from __future__ import print_function

import argparse
import time

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from decoder import GreedyDecoder, GreedyDecoderMaxOffset

from data.data_loader import SpectrogramDataset, AudioDataLoader
from model import DeepSpeech



def filter_usable_words(s):
    # remove leading whitespace since this would mess up our metric
    # to take the second *word* in the prediction.
    s = s.strip()
    s = s.split(' ')
    if len(s) < 3:
        return ''
    return s[1:-1]

def pp_joint(out, p):
    s = out[0][0]
    p = p.tolist()

    o1 = "".join(["{:>4} ".format(n) for n in s])
    o2 = "".join(["{:.2f} ".format(n) for n in p])

    return "\n".join([o1, o2])


beam_alpha = 2.15
beam_beta = 0.35
beam_size = 500
cutoff_prob = 1.0
cutoff_top_n = 40


def external_decoder(vocab_list, scorer, infer_results):
    from swig_decoders import ctc_beam_search_decoder_batch
    scorer.reset_params(beam_alpha, beam_beta)

    # TODO: is dis gud?
    probs_split = infer_results
    print(probs_split.shape)
    probs_list = probs_split.numpy().tolist()

    print(probs_list, type(probs_list[0][0][0]))

    # beam search decode
    num_processes = 1 # min(num_processes, len(probs_split))
    beam_search_results = ctc_beam_search_decoder_batch(
        probs_split,
        vocab_list,
        beam_size,
        num_processes,
        cutoff_prob,
        cutoff_top_n,
        scorer,
    )

    results = [result[0][1] for result in beam_search_results]
    return results


def setup_scorer(language_model_path, vocab_list):
    from swig_decoders import Scorer
    print("begin to initialize the external scorer "
                     "for decoding")
    _ext_scorer = Scorer(beam_alpha, beam_beta, language_model_path, vocab_list)

    lm_char_based = _ext_scorer.is_character_based()
    lm_max_order = _ext_scorer.get_max_order()
    lm_dict_size = _ext_scorer.get_dict_size()
    print("language model: "
                     "is_character_based = %d," % lm_char_based +
                     " max_order = %d," % lm_max_order +
                     " dict_size = %d" % lm_dict_size)
    print("end initializing scorer. Start decoding ...")

    return _ext_scorer


def transcribe(model, language_model_path, decoder, q):
    hidden = None
    accoustic_data = []
    a_data_fac = 4

    vocab_list = DeepSpeech.get_labels(model)
    vocab_list = [chars.encode("utf-8") for chars in vocab_list]

    scorer = setup_scorer(language_model_path, vocab_list)

    while True:
        step, spect = q.get()

        tick = time.time()

        spect = torch.from_numpy(spect)

        print("modeling step", step)

        spect_in = spect.contiguous().view(1, 1, spect.size(0), spect.size(1))
        spect_in = torch.autograd.Variable(spect_in, volatile=True)
        out, hidden = model(spect_in, hidden)
        out = out.transpose(0, 1)  # TxNxH

        print('od',out.data.shape)

        accoustic_data.append(out.data)

        if len(accoustic_data) > a_data_fac:
            accoustic_data.pop(0)

        if len(accoustic_data) < a_data_fac:
            continue

        buffered_probs = torch.cat(accoustic_data, dim=0)

        if True:
            external_decoder(vocab_list, scorer, buffered_probs)
        elif isinstance(decoder, GreedyDecoderMaxOffset):
            decoded_output, offsets, cprobs = decoder.decode(buffered_probs)
            pp = pp_joint(decoded_output, cprobs)
            print(pp)
        else:
            decoded_output, offsets = decoder.decode(buffered_probs)
            print(decoded_output)

        tock = time.time()
        print("model time:", tock - tick)


if __name__ == '__main__':
    import argparse
    from multiprocessing import Queue, Process

    import capture

    parser = argparse.ArgumentParser(description='DeepSpeech transcription')
    parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                        help='Path to model file created by training')
    parser.add_argument('--test_manifest', metavar='DIR',
                        help='path to validation manifest csv', default='data/test_manifest.csv')
    parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
    parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam", "pp"], type=str, help="Decoder to use")
    parser.add_argument('--padding_t', default=10, type=int)
    parser.add_argument('--use_file', action='store_true')

    beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
    beam_args.add_argument('--top_paths', default=1, type=int, help='number of beams to return')
    beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
    beam_args.add_argument('--lm_path', default=None, type=str,
                           help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
    beam_args.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
    beam_args.add_argument('--beta', default=1, type=float, help='Language model word bonus (all words)')
    beam_args.add_argument('--cutoff_top_n', default=40, type=int,
                           help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                                'vocabulary will be used in beam search, default 40.')
    beam_args.add_argument('--cutoff_prob', default=1.0, type=float,
                           help='Cutoff probability in pruning,default 1.0, no pruning.')
    beam_args.add_argument('--lm_workers', default=1, type=int, help='Number of LM processes to use')
    args = parser.parse_args()

    model = DeepSpeech.load_model(args.model_path, padding_t=args.padding_t)
    model.eval()

    audio_conf = DeepSpeech.get_audio_conf(model)
    labels = DeepSpeech.get_labels(model)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder
        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    elif args.decoder == "pp":
        decoder = None
    else:
        decoder = GreedyDecoderMaxOffset(labels, blank_index=labels.index('_'))

    q = Queue()

    p_capture = Process(target=capture.capture, args=(audio_conf, args.use_file, q,))
    p_transcribe = Process(target=transcribe, args=(model, args.lm_path, decoder, q,))

    try:
        p_capture.start()
        p_transcribe.start()

        p_capture.join()
        p_transcribe.join()
    except KeyboardInterrupt:
        p_capture.terminate()
        p_transcribe.terminate()
