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


def transcribe(model, q, lm_q):
    hidden = None

    while True:
        step, spect = q.get()

        tick = time.time()

        spect = torch.from_numpy(spect)

        print("modeling step", step)

        spect_in = spect.contiguous().view(1, 1, spect.size(0), spect.size(1))
        spect_in = torch.autograd.Variable(spect_in, volatile=True)
        out, hidden = model(spect_in, hidden)
        out = out.transpose(0, 1)  # TxNxH

        print('od', out.data.shape)

        lm_q.put((step, out))

        tock = time.time()
        print("model time:", tock - tick)



def language_model(model, decoder, language_model_path, q):
    accoustic_data = []
    a_data_fac = 3

    while True:
        (step, out) = q.get()

        print("language model step", step)

        accoustic_data.append(out.data)

        if len(accoustic_data) > a_data_fac:
            accoustic_data.pop(0)

        if len(accoustic_data) < a_data_fac:
            continue

        buffered_probs = torch.cat(accoustic_data, dim=0)

        if decoder == "pp":
            results = decoder.decode(buffered_probs)
            print(results)
        elif isinstance(decoder, GreedyDecoderMaxOffset):
            decoded_output, offsets, cprobs = decoder.decode(buffered_probs)
            pp = pp_joint(decoded_output, cprobs)
            print(pp)
        else:
            decoded_output, offsets = decoder.decode(buffered_probs)
            print(decoded_output)

        assert False




class PPBeamScorer:
    def __init__(self, alphabet, language_model_path=None, scorer_vocab_path=None):
        self.beam_alpha = 2.15
        self.beam_beta = 0.35
        self.beam_size = 500
        self.cutoff_prob = 1.0
        self.cutoff_top_n = 40
        self.num_processes = 1 # min(num_processes, len(probs_split))
        self.language_model_path = language_model_path
        self.scorer_vocab_path = scorer_vocab_path
        self.alphabet = alphabet

        if self.language_model_path:
            assert self.scorer_vocab_path is not None, "Supply a vocabulary (words) path"
            self.vocab_list = self.get_vocab_list(scorer_vocab_path)
            self.scorer = self.setup_scorer(
                    self.language_model_path,
                    self.get_vocab_list(scorer_vocab_path)
                )
        else:
            self.vocab_list = None
            self.scorer = None

    def get_vocab_list(self, path):
        with open(path, 'r') as f:
            vocab_list = f.readlines()
        vocab_list = [chars.strip().encode("utf-8") for chars in vocab_list]
        print(vocab_list[:100])
        return vocab_list

    def setup_scorer(self, language_model_path, vocab_list):
        from swig_decoders import Scorer
        print("begin to initialize the external scorer for decoding")
        print("alpha = {}, beta = {}, lm_path = {}, len(vocab_list) = {}".format(
            self.beam_alpha, self.beam_beta, language_model_path, len(vocab_list)))

        _ext_scorer = Scorer(self.beam_alpha, self.beam_beta, language_model_path, vocab_list)

        lm_char_based = _ext_scorer.is_character_based()
        lm_max_order = _ext_scorer.get_max_order()
        lm_dict_size = _ext_scorer.get_dict_size()
        print("language model: "
                         "is_character_based = %d," % lm_char_based +
                         " max_order = %d," % lm_max_order +
                         " dict_size = %d" % lm_dict_size)
        print("end initializing scorer. Start decoding ...")

        return _ext_scorer

    # TODO: vocab_list must be alphabet, infer_results must be sorted?
    def decode(self, infer_results):
        from swig_decoders import ctc_beam_search_decoder_batch

        # PP beam decoder expects blank to be at last index, everyone
        # else has the blank at idx=0, we swap them here.
        blanks = infer_results[:, :, 0]
        lasts = infer_results[:, :, -1]
        infer_results[:, :, -1] = blanks
        infer_results[:, :, 0] = lasts

        probs_split = infer_results.transpose(0, 1)
        assert len(self.alphabet) + 1 == probs_split.size(-1)

        probs_list = probs_split.numpy().tolist()
        assert type(probs_list[0][0][0]) == float

        # beam search decode
        if self.scorer is None:
            beam_search_results = ctc_beam_search_decoder_batch(
                probs_list,
                self.alphabet,
                self.beam_size,
                self.num_processes,
                self.cutoff_prob,
                self.cutoff_top_n,
        #        scorer,
            )
        else:
            self.scorer.reset_params(self.beam_alpha, self.beam_beta)
            beam_search_results = ctc_beam_search_decoder_batch(
                probs_list,
                self.alphabet,
                self.beam_size,
                self.num_processes,
                self.cutoff_prob,
                self.cutoff_top_n,
                self.scorer,
            )
        print("foo", beam_search_results)

        results = [result[0][1] for result in beam_search_results]
        return results


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
    beam_args.add_argument('--scorer_vocab_path', default='words.txt', type=str,
                           help='Path to vocab file')
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
        # the following removes the blank char from the vocabulary and
        # puts the last index (assumed by PP to be blank index) to the beginning
        # (beginning is assumed by everyone else to be blank (idx=0)).
        # The vocab then looks like this:
        #
        #    alphabet = " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        #
        labels = labels[-1] + labels[1:-1]
        print('"%s"' % labels, len(labels))
        labels = [chars.encode("utf-8") for chars in labels]

        decoder = PPBeamScorer(
                labels,
                language_model_path=args.lm_path,
                scorer_vocab_path=args.scorer_vocab_path)
    else:
        decoder = GreedyDecoderMaxOffset(labels, blank_index=labels.index('_'))

    q = Queue()
    lm_q = Queue()

    p_capture = Process(target=capture.capture, args=(audio_conf, args.use_file, q,))
    p_transcribe = Process(target=transcribe, args=(model, q, lm_q,))
    p_model = Process(target=language_model, args=(model, decoder, args.lm_path, lm_q,))

    try:
        p_capture.start()
        p_transcribe.start()
        p_model.start()

        p_capture.join()
        p_transcribe.join()
        p_model.join()
    except KeyboardInterrupt:
        p_capture.terminate()
        p_transcribe.terminate()
        p_model.terminate()
