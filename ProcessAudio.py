import os
import shutil

import librosa
import librosa.core as lb
import librosa.display as lbd
import matplotlib.pyplot as plt
import numpy
import pyloudnorm as pyln
import soundfile as sf
import torch
from torchaudio.transforms import MFCC, MuLawEncoding, MuLawDecoding, Resample
from torchaudio.transforms import Vad as VoiceActivityDetection


class AudioPreprocessor:
    def __init__(self, sr, new_sr=None, n_mfccs=120):
        self.sr = sr
        self.new_sr = new_sr
        self.vad = VoiceActivityDetection(sample_rate=sr)
        self.mu_decode = MuLawDecoding()
        self.mu_encode = MuLawEncoding()
        self.meter = pyln.Meter(sr)
        self.mfcc = MFCC(sample_rate=self.sr, n_mfcc=n_mfccs)
        if new_sr is not None:
            self.resample = Resample(orig_freq=sr, new_freq=new_sr)
        else:
            self.resample = lambda x: x

    def apply_mu_law(self, audio):
        """
        encodes the signal and then
        decodes it. It is lossy, but
        the idea is that the lossyness
        only affects the noise and the
        voice is retained.
        """
        return self.mu_decode(self.mu_encode(audio))

    def cut_silence_from_beginning(self, audio):
        """
        applies cepstral voice activity
        detection and noise reduction to
        cut silence from the beginning of
        a recording
        """
        return self.vad(torch.from_numpy(audio))

    def to_mono(self, x):
        """
        make sure we deal with a 1D array
        """
        if len(x.shape) == 2:
            return lb.to_mono(numpy.transpose(x))
        else:
            return x

    def normalize_loudness(self, audio):
        """
        normalize the amplitudes according to
        their decibels, so this should turn any
        signal with different magnitudes into
        the same magnitude by analysing loudness
        """

        loudness = self.meter.integrated_loudness(audio)
        return pyln.normalize.loudness(audio, loudness, -30.0)

    def process_audio(self, audio):
        """
        one function to apply them all in an
        order that makes sense.
        """
        audio = self.to_mono(audio)
        audio = self.normalize_loudness(audio)
        audio = self.cut_silence_from_beginning(audio)
        audio = self.apply_mu_law(audio)
        audio = self.resample(audio)
        return audio

    def to_mfcc(self, audio, normalize=True):
        """
        outputs a matrix of MFCCs
        """
        if normalize:
            audio = self.process_audio(audio)
        else:
            audio = torch.tensor(audio)
        return self.mfcc(audio)

    def visualize_cleaning(self, unclean_audio):
        """
        displays Mel Spectrogram of unclean audio
        and then displays Mel Spectrogram of the
        cleaned version.
        """
        fig, ax = plt.subplots(nrows=2, ncols=1)
        unclean_audio = self.to_mono(unclean_audio)
        clean_audio = numpy.array(self.process_audio(unclean_audio))
        unclean = numpy.log(librosa.feature.melspectrogram(y=unclean_audio, sr=self.sr, power=1))
        clean = numpy.log(librosa.feature.melspectrogram(y=clean_audio, sr=self.sr, power=1))
        lbd.specshow(unclean, sr=self.sr, cmap='GnBu', y_axis='mel', ax=ax[0], x_axis=None)
        ax[0].set(title='Uncleaned Audio')
        ax[0].label_outer()
        if self.new_sr is not None:
            lbd.specshow(clean, sr=self.new_sr, cmap='GnBu', y_axis='mel', ax=ax[1], x_axis=None)
        else:
            lbd.specshow(clean, sr=self.sr, cmap='GnBu', y_axis='mel', ax=ax[1], x_axis=None)
        ax[1].set(title='Cleaned Audio')
        ax[1].label_outer()
        plt.show()


def read(path):
    return sf.read(path, dtype='float32')


def write(path, audio, sr):
    os.remove(path)
    sf.write(path, audio, sr)


def normalize_corpus(path_to_orig_corpus, path_to_normalized_clone, desired_sr=None):
    """
    prepares an entire corpus at once
    """
    # structure has to be: path_to_orig_corpus/speakerID/audios/filename.wav
    # everything else doesn't matter and is kept the same

    # clone the corpus
    if os.path.exists(path_to_normalized_clone):
        shutil.rmtree(path_to_normalized_clone)
    shutil.copytree(path_to_orig_corpus, path_to_normalized_clone)

    # go through clone and process the audios
    audio_preprocessor = None
    for speakerID in os.listdir(path_to_normalized_clone):
        for audio in os.listdir(os.path.join(path_to_normalized_clone, speakerID, "audios")):
            print("Processing audio {} from speaker {}".format(audio, speakerID))
            raw_audio, sr = read(os.path.join(path_to_normalized_clone, speakerID, "audios", audio))
            if audio_preprocessor is None:
                audio_preprocessor = AudioPreprocessor(sr=sr, new_sr=desired_sr)
                if not desired_sr:
                    desired_sr = sr
            processed_audio = audio_preprocessor.process_audio(raw_audio)
            write(os.path.join(path_to_normalized_clone, speakerID, "audios", audio), processed_audio, desired_sr)


if __name__ == '__main__':
    np, fs = read("test_corp/Flux/audios/test.wav")
    ap = AudioPreprocessor(sr=fs)
    ap.visualize_cleaning(np)
