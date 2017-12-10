import argparse

import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

from decoder import GreedyDecoder, GreedyDecoderMaxOffset

from data.data_loader import SpectrogramDataset, AudioDataLoader
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam", "none"], type=str, help="Decoder to use")
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
no_decoder_args = parser.add_argument_group("No Decoder Options", "Configuration options for when no decoder is "
                                                                  "specified")
no_decoder_args.add_argument('--output_path', default=None, type=str, help="Where to save raw acoustic output")
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

if __name__ == '__main__':
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    decoder = GreedyDecoderMaxOffset(labels, blank_index=labels.index('_'))
    target_decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.test_manifest, labels=labels,
                                      normalize=True)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)


    import torch
    import librosa
    import alsaaudio
    import librosa.display

    sample_rate = audio_conf['sample_rate']
    window_stride = audio_conf['window_stride']
    window_size = audio_conf['window_size']
    window = audio_conf['window']
    normalize = True

    window_size_abs = int(window_size * sample_rate)

    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
    inp.setchannels(1)
    inp.setrate(audio_conf['sample_rate'])
    inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)

    print(inp.pcmmode())

    #inp.setperiodsize(window_size_abs)

    import visdom
    vis = visdom.Visdom()

    import torchaudio
    sound, _ = torchaudio.load('samples/SA1.WAV')
    sound = sound.numpy()[:,0]

    n = 11
    idxs = zip(range(0, len(sound), n), range(0, len(sound), n)[1:])
    img = np.zeros((11*4, window_size_abs))
    img_i = 0
    h = None

    inp.setperiodsize(n)

    print(float(np.prod(img.shape)) / sample_rate, "seconds of audio buffered")

    #for start, end in idxs:
    #    y = sound[start*window_size_abs:end*window_size_abs]
    while True:
        l, data = inp.read()
        y = np.fromstring(data, dtype='int16')

        print("y.shape",y.shape)

        img[:-n] = img[n:]
        img[-n:] = y.reshape(n, -1)
        y = img.flatten()

        """
        img[:-n] = img[n:]
        foo = y.reshape((n, -1))
        img[-n:] = foo

        y = img.flatten()
        """

        # pre-fill buffer
        if img_i < img.shape[0]:
            img_i += n
            continue

        n_fft = int(sample_rate * window_size)
        win_length = n_fft
        hop_length = int(sample_rate * window_stride)

        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=window)
        spect, phase = librosa.magphase(D)

        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        #librosa.display.specshow(spect)
        #vis.image(spect.numpy(), win="foo")
        #print(spect.size())

        spect_in = spect.contiguous().view(1, 1, spect.size(0), spect.size(1))
        spect_in = torch.autograd.Variable(spect_in, volatile=True)
        out, h = model(spect_in, h)
        out = out.transpose(0, 1)  # TxNxH

        print("out.size", out.size())

        decoded_output, offsets, cprobs = decoder.decode(out.data)
#        print(filter_usable_words(decoded_output[0][0]))

        #print("dout", decoded_output)
        #print("probs", cprobs)

        pp = pp_joint(decoded_output, cprobs)
        print(pp)




    assert False

    output_data = []
    h = None
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = data

        print(inputs)

        inputs = Variable(inputs, volatile=True)

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        if args.cuda:
            inputs = inputs.cuda()

        out, h = model(inputs, h)
        print(h)

        out = out.transpose(0, 1)  # TxNxH
        seq_length = out.size(0)
        sizes = input_percentages.mul_(int(seq_length)).int()

        if decoder is None:
            # add output to data array, and continue
            output_data.append((out.data.cpu().numpy(), sizes.numpy()))
            continue

        decoded_output, _, = decoder.decode(out.data, sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)

        print(decoded_output)

        wer, cer = 0, 0
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(transcript, reference) / float(len(reference.split()))
            cer_inst = decoder.cer(transcript, reference) / float(len(reference))
            wer += wer_inst
            cer += cer_inst
            if args.verbose:
                print("Ref:", reference.lower())
                print("Hyp:", transcript.lower())
                print("WER:", wer_inst, "CER:", cer_inst, "\n")

    np.save(args.output_path, output_data)
