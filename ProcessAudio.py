import os
import shutil

import librosa.core as lb
import numpy
import pyloudnorm as pyln
import soundfile as sf
import torch
from scipy.signal import butter, sosfilt
from torchaudio.compliance.kaldi import fbank
from torchaudio.transforms import Vad as VoiceActivityDetection


class AudioPreprocessor:
    def __init__(self, sr):
        self.sr = sr
        self.vad = VoiceActivityDetection(sample_rate=sr)  # TODO The settings need tweaking

    def cut_silence_from_beginning(self, np):
        """
        applies cepstral voice activity
        detection and noise reduction to
        cut silence from the beginning of
        a recording
        """
        return numpy.array(self.vad.forward(torch.from_numpy(np)))

    def to_mono(self, x):
        """
        make sure we deal with a 1D array
        """
        if len(x.shape) == 2:
            return lb.to_mono(numpy.transpose(x))
        else:
            return x

    def resample(self, x, new_sr):
        """
        change sampling rate, if so desired
        """
        if self.sr != new_sr:
            return lb.resample(x, self.sr, new_sr)
        else:
            return x

    def apply_mu_law(self, x):
        """
        if I understand this right: this
        compresses the signal with the goal
        of preserving speech, but the lossyness
        takes out non-speech (--> noise)
        """
        return lb.mu_expand(lb.mu_compress(x))

    def normalize_loudness(self, np):
        """
        normalize the amplitudes according to
        their decibels, so this should turn any
        signal with different magnitudes into
        the same magnitude by analysing loudness
        """
        meter = pyln.Meter(self.sr)
        loudness = meter.integrated_loudness(np)
        return pyln.normalize.loudness(np, loudness, -30.0)

    def apply_bandpass_filter(self, np, lowcut=50.0, highcut=8000.0, order=5):
        """
        filter out frequencies above 8000Hz and
        below 50Hz since those are outside the
        human voice range and thus likely noise
        """
        nyq = 0.5 * self.sr
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        y = sosfilt(sos, np)
        return y

    def process_audio(self, np, desired_sr=None):
        """
        one function to apply them all in the
        correct order. To skip a step, simply
        comment out the line.
        """
        audio = self.to_mono(np)
        audio = self.normalize_loudness(audio)  # in here twice, once that we get consistent input levels to the VAD
        audio = self.cut_silence_from_beginning(audio)
        audio = self.apply_mu_law(audio)
        audio = self.apply_bandpass_filter(audio)
        if desired_sr is not None:
            audio = self.resample(audio, desired_sr)
        audio = self.normalize_loudness(audio)  # and once to make sure the final result retained scale
        return audio

    def to_mfcc(self, np):
        """
        outputs a spectrogram as tensor for a
        given array
        """
        normalized_audio = self.process_audio(np)
        return audio_to_mfcc_tensor(normalized_audio, self.sr)


def read(path):
    return sf.read(path, dtype='float32')


def write(path, audio, sr):
    os.remove(path)
    sf.write(path, audio, sr)


def audio_to_mfcc_tensor(audio, sr):
    audio_unsqueezed = torch.from_numpy(audio).unsqueeze(0)
    filter_bank = fbank(audio_unsqueezed, sample_frequency=sr, num_mel_bins=80)
    pitch = torch.zeros(filter_bank.shape[0], 3)
    speech_in_features = torch.cat([filter_bank, pitch], 1).numpy()
    return speech_in_features


def normalize_corpus(path_to_orig_corpus, path_to_normalized_clone, desired_sr=None):
    """
    prepares an entire corpus at once
    """
    # structure has to be: path_to_orig_corpus/speakerID/audios/filename.wav
    # everything else doesn't matter and is kept the same

    audio_preprocessor = None

    # set up the clone
    if os.path.exists(path_to_normalized_clone):
        shutil.rmtree(path_to_normalized_clone)
    shutil.copytree(path_to_orig_corpus, path_to_normalized_clone)

    # go through clone and process the audios
    for speakerID in os.listdir(path_to_normalized_clone):
        for audio in os.listdir(os.path.join(path_to_normalized_clone, speakerID, "audios")):
            print("Processing audio {} from speaker {}".format(audio, speakerID))
            raw_audio, sr = read(os.path.join(path_to_normalized_clone, speakerID, "audios", audio))
            if audio_preprocessor is None:
                audio_preprocessor = AudioPreprocessor(sr=sr)
            if desired_sr is None:
                desired_sr = sr
            processed_audio = audio_preprocessor.process_audio(raw_audio, desired_sr)
            write(os.path.join(path_to_normalized_clone, speakerID, "audios", audio), processed_audio, desired_sr)


if __name__ == '__main__':
    # test
    normalize_corpus("test_corp", "test_corp_norm")
