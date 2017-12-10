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


def transcribe(model, q):
    labels = DeepSpeech.get_labels(model)
    decoder = GreedyDecoderMaxOffset(labels, blank_index=labels.index('_'))
    hidden = None

    while True:
        spect = q.get()

        tick = time.time()
        #vis.image(spect.numpy(), win="foo")

        spect = torch.from_numpy(spect)
        spect_in = spect.contiguous().view(1, 1, spect.size(0), spect.size(1))
        spect_in = torch.autograd.Variable(spect_in, volatile=True)
        out, hidden = model(spect_in, hidden)
        out = out.transpose(0, 1)  # TxNxH

        print("out.size", out.size())

        decoded_output, offsets, cprobs = decoder.decode(out.data)
#        print(filter_usable_words(decoded_output[0][0]))
        #print("dout", decoded_output)
        #print("probs", cprobs)

        pp = pp_joint(decoded_output, cprobs)
        print(pp)

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

    model = DeepSpeech.load_model(args.model_path)
    model.eval()

    audio_conf = DeepSpeech.get_audio_conf(model)


    q = Queue()

    p_capture = Process(target=capture.capture, args=(audio_conf, q,))
    p_transcribe = Process(target=transcribe, args=(model, q,))

    p_capture.start()
    p_transcribe.start()

    p_capture.join()
    p_transcribe.join()
