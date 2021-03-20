import re
import sys
from collections import defaultdict

import numpy
import phonemizer
import torch
from cleantext import clean


class TextFrontend:
    def __init__(self,
                 language,
                 use_panphon_vectors=True,
                 use_word_boundaries=False,
                 use_explicit_eos=False):
        """
        Mostly preparing ID lookups
        """
        self.use_panphon_vectors = use_panphon_vectors
        self.use_word_boundaries = use_word_boundaries
        self.use_explicit_eos = use_explicit_eos

        # list taken and modified from https://github.com/dmort27/panphon
        # see publication: https://www.aclweb.org/anthology/C16-1328/
        self.ipa_to_vector = defaultdict()
        if use_panphon_vectors:
            self.default_vector = [132, 132, 132, 132, 132, 132, 132, 132, 132, 132,
                                   132, 132, 132, 132, 132, 132, 132, 132, 132, 132,
                                   132, 132, 132, 132, 132]
        else:
            self.default_vector = 132
        with open("PreprocessingForTTS/ipa_vector_lookup.csv", encoding='utf8') as f:
            features = f.read()
        features_list = features.split("\n")
        for index in range(1, len(features_list)):
            line_list = features_list[index].split(",")
            if use_panphon_vectors:
                self.ipa_to_vector[line_list[0]] = [float(x) for x in line_list[1:]]
            else:
                self.ipa_to_vector[line_list[0]] = index
                # note: Index 0 is unused, so it can be used for padding as is convention.
                #       Index 1 is reserved for EOS, if you want to use explicit EOS.
                #       Index 132 is used for unknown characters
                #       Index 10 is used for pauses (heuristically)

        if language == "en":
            self.clean_lang = "en"
            self.g2p_lang = "en-us"

        elif language == "de":
            self.clean_lang = "de"
            self.g2p_lang = "de"

        else:
            print("Language not supported yet")
            sys.exit()

    def string_to_tensor(self, text, view=False):
        """
        applies the entire pipeline to a text
        """
        # clean unicode errors etc
        utt = clean(text, fix_unicode=True, to_ascii=True, lower=False, lang=self.clean_lang)

        if self.clean_lang == "en":
            utt = english_text_expansion(utt)

        # phonemize
        phones = phonemizer.phonemize(utt,
                                      language_switch='remove-flags',
                                      backend="espeak",
                                      language=self.g2p_lang,
                                      preserve_punctuation=True,
                                      strip=True,
                                      with_stress=True).replace(";", ",").replace(":", ",").replace('"', ",").replace(
            "--", ",").replace("\n", " ").replace("\t", " ")
        if view:
            print("Phonemes: \n{}\n".format(phones))

        tensors = list()
        phones_vector = list()

        # turn into numeric vectors
        for char in phones:
            if self.use_word_boundaries:
                if char != " ":
                    phones_vector.append(self.ipa_to_vector.get(char, self.default_vector))
            else:
                phones_vector.append(self.ipa_to_vector.get(char, self.default_vector))

        if self.use_explicit_eos:
            phones_vector.append(self.ipa_to_vector["end_of_input"])

        # turn into tensors
        if self.use_panphon_vectors:
            for line in numpy.transpose(numpy.array(phones_vector)):
                tensors.append(torch.LongTensor(line))
        else:
            tensors.append(torch.LongTensor(phones_vector))

        # combine tensors and return
        return torch.stack(tensors, 0)


def english_text_expansion(text):
    """
    Apply as small part of the tacotron style text cleaning pipeline, suitable for e.g. LJSpeech.
    See https://github.com/keithito/tacotron/
    Careful: Only apply to english datasets. Different languages need different cleaners.
    """
    _abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
        ('Mrs.', 'misess'),
        ('Mr.', 'mister'),
        ('Dr.', 'doctor'),
        ('St.', 'saint'),
        ('Co.', 'company'),
        ('Jr.', 'junior'),
        ('Maj.', 'major'),
        ('Gen.', 'general'),
        ('Drs.', 'doctors'),
        ('Rev.', 'reverend'),
        ('Lt.', 'lieutenant'),
        ('Hon.', 'honorable'),
        ('Sgt.', 'sergeant'),
        ('Capt.', 'captain'),
        ('Esq.', 'esquire'),
        ('Ltd.', 'limited'),
        ('Col.', 'colonel'),
        ('Ft.', 'fort')]]
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


if __name__ == '__main__':
    # test an English utterance
    tfr_en = TextFrontend(language="en",
                          use_panphon_vectors=False,
                          use_word_boundaries=False,
                          use_explicit_eos=False)
    print(tfr_en.string_to_tensor("Hello!"))

    # test a German utterance
    tfr_de = TextFrontend(language="de",
                          use_panphon_vectors=True,
                          use_word_boundaries=True,
                          use_explicit_eos=True)
    print(tfr_de.string_to_tensor("Hallo!"))
