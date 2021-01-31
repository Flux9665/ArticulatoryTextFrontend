import librosa
import librosa.core as lb
import librosa.display as lbd
import matplotlib.pyplot as plt
import numpy
import pyloudnorm as pyln
import soundfile as sf
import torch
from torchaudio.functional import mu_law_decoding, mu_law_encoding
from torchaudio.transforms import MFCC, MuLawEncoding, Resample
from torchaudio.transforms import Vad as VoiceActivityDetection


class AudioPreprocessor:
    def __init__(self, input_sr, output_sr=None, n_mfccs=120):
        self.sr = input_sr
        self.new_sr = output_sr
        self.vad = VoiceActivityDetection(sample_rate=input_sr)
        self.mu_encode = MuLawEncoding()
        self.meter = pyln.Meter(input_sr)
        self.mfcc = MFCC(sample_rate=self.sr, n_mfcc=n_mfccs)
        self.final_sr = input_sr
        if output_sr is not None:
            self.resample = Resample(orig_freq=input_sr, new_freq=output_sr)
            self.final_sr = output_sr
        else:
            self.resample = lambda x: x

    def apply_mu_law(self, audio):
        """
        brings the audio down from 16 bit
        resolution to 8 bit resolution to
        make using softmax to predict a
        wave from it more feasible.

        !CAREFUL! transforms the floats
        between -1 and 1 to integers
        between 0 and 255. So that is good
        to work with, but bad to save/listen
        to. Apply mu-law decoding before
        saving or listening to the audio.
        """
        return self.mu_encode(audio)

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
        audio = self.resample(audio)
        audio = self.apply_mu_law(audio)
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
        clean_audio = numpy.array(self.process_audio(unclean_audio), dtype='float32')

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


if __name__ == '__main__':
    wave, fs = sf.read("test_audio/test.wav")
    ap = AudioPreprocessor(input_sr=fs, output_sr=16000)
    ap.visualize_cleaning(wave)

    clean_wave_mulaw = ap.process_audio(wave)
    sf.write("test_audio/test_cleaned.wav", mu_law_decoding(clean_wave_mulaw, quantization_channels=256), ap.final_sr)

    print(clean_wave_mulaw)
