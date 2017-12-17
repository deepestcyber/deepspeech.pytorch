from __future__ import print_function

import numpy as np
import librosa
import alsaaudio


def infinite_loop():
    while True:
        yield (0,0)


def capture(audio_conf, use_file, queue):
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

    BUFFER_SECONDS = 1
    buffer_dim = np.ceil(float(BUFFER_SECONDS * sample_rate) / window_size_abs)
    buffer_dim = int(buffer_dim)
    buffer_fill_level = buffer_dim / 2

    import torchaudio
    sound, _ = torchaudio.load('samples/SA1.WAV')
    sound = sound.numpy()[:,0]

    pad_len = np.ceil(float(len(sound)) / window_size_abs) * window_size_abs
    sound_padded = np.zeros(int(pad_len), dtype=sound.dtype)
    sound_padded[:len(sound)] = sound
    sound = sound_padded

    if use_file:
        buffer_dtype = sound.dtype
    else:
        buffer_dtype = 'int16'

    # window to send to model
    k = None
    # feed
    n = 11
    if use_file:
        idxs = zip(range(0, len(sound), n), range(0, len(sound), n)[1:])
    else:
        idxs = infinite_loop()
    img = np.zeros((buffer_dim, window_size_abs), dtype=buffer_dtype)
    img_i = 0

    inp.setperiodsize(n * window_size_abs)

    print(float(np.prod(img.shape)) / sample_rate, "seconds of audio buffered")

    import wave
    write_to_file = False
    debug_i = 0
    step = 0


    if write_to_file:
        f = wave.open('debug/sample_{}.wav'.format(debug_i), mode='wb')
        f.setparams((1, 2, sample_rate, 0, 'NONE', 'not compressed'))


    def compute_spect(y, eps=1e-6):
        n_fft = int(sample_rate * window_size)
        win_length = n_fft
        hop_length = int(sample_rate * window_stride)

        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=window)
        spect, phase = librosa.magphase(D)

        # S = log(S+1)
        spect = np.log1p(spect)
        if normalize:
            mean = spect.mean()
            std = spect.std()
            spect -= mean
            spect /= std + eps
        return spect


    for start, end in idxs:
        if use_file:
            y = sound[start*window_size_abs:end*window_size_abs]
        else:
            l, data = inp.read()
            y = np.fromstring(data, dtype='int16')

        img[:-n] = img[n:]
        img[-n:] = y.reshape((n, -1))
        y = img[:k].flatten()

        # fill buffer until point-of-send
        if img_i < buffer_fill_level:
            img_i += n
            continue

        if write_to_file:
            f.writeframes(y.tostring())
            debug_i += 1

        spect = compute_spect(y)
        print("queueing step", step)
        queue.put((step, spect))

        step += 1
        img_i = 0

    # Flush in case of file
    queue.put(compute_spect(img.flatten()))

    print("Finished capture")

if __name__ == "__main__":
    import argparse
    from multiprocessing import Queue

    from model import DeepSpeech

    parser = argparse.ArgumentParser(description='Audio capture')
    parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                        help='Path to model file created by training')
    args = parser.parse_args()

    model = DeepSpeech.load_model(args.model_path)
    model.eval()

    audio_conf = DeepSpeech.get_audio_conf(model)

    q = Queue()
    capture(audio_conf, q)

