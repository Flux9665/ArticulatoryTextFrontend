import sys
from collections import defaultdict

import numpy
import phonemizer
import spacy
import torch
from cleantext import clean

"""
Dimensions in a Tensor correspond to:

syl (binary, phonetic property of phoneme)
son (binary, phonetic property of phoneme)
cons (binary, phonetic property of phoneme)
cont (binary, phonetic property of phoneme)
delrel (binary, phonetic property of phoneme)
lat (binary, phonetic property of phoneme)
nas (binary, phonetic property of phoneme)
strid (binary, phonetic property of phoneme)
voi (binary, phonetic property of phoneme)
sg (binary, phonetic property of phoneme)
cg (binary, phonetic property of phoneme)
ant (binary, phonetic property of phoneme)
cor (binary, phonetic property of phoneme)
distr (binary, phonetic property of phoneme)
lab (binary, phonetic property of phoneme)
hi (binary, phonetic property of phoneme)
lo (binary, phonetic property of phoneme)
back (binary, phonetic property of phoneme)
round (binary, phonetic property of phoneme)
velaric (binary, phonetic property of phoneme)
tense (binary, phonetic property of phoneme)
long (binary, phonetic property of phoneme)
hitone (binary, phonetic property of phoneme)
hireg (binary, phonetic property of phoneme)

stress (binary, stressmarker identity)

pos (1 = content word, 2 = function word, 3 = other word, 0 = space)

position in sequence (float, corresponds to percent of sequence from left to right)
"""


class TextFrontend:
    def __init__(self, language, use_shallow_pos=True, use_positional_information=True):
        """
        Mostly loading the right spacy
        models and preparing ID lookups
        """
        self.use_shallow_pos = use_shallow_pos
        self.use_positional_information = use_positional_information

        # list taken and modified from https://github.com/dmort27/panphon
        self.ipa_to_vector = defaultdict()
        self.default_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        with open("ipa_bases.csv", encoding='utf8') as f:
            features = f.read()
        features_list = features.split("\n")
        for index in range(1, len(features_list)):
            line_list = features_list[index].split(",")
            self.ipa_to_vector[line_list[0]] = [float(x) for x in line_list[1:]]

        if language == "en":
            self.clean_lang = "en"
            self.g2p_lang = "en-us"
            self.nlp = spacy.load('en_core_web_sm')

        elif language == "de":
            self.clean_lang = "de"
            self.g2p_lang = "de"
            self.nlp = spacy.load('de_core_news_sm')

        else:
            print("Language not supported yet")
            sys.exit()

        self.tokenizer = self.nlp.Defaults.create_tokenizer(self.nlp)

        if self.use_shallow_pos:
            self.tagger = self.nlp.get_pipe("tagger")
            content_word_tags = {"ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"}
            function_word_tags = {"ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"}
            other_tags = {"PUNCT", "SYM", "X"}
            self.tag_id_lookup = {"SPACE__": 0}
            for tag in content_word_tags:
                self.tag_id_lookup[tag] = 1
            for tag in function_word_tags:
                self.tag_id_lookup[tag] = 2
            for tag in other_tags:
                self.tag_id_lookup[tag] = 3

    def string_to_tensor(self, text, view=False):
        """
        applies the entire pipeline to a text
        """
        # tokenize
        utt = self.tokenizer(text)

        # clean emojis etc
        cleaned_tokens = []
        for index, token in enumerate(utt):
            cleaned_tokens.append(clean(token, fix_unicode=True, to_ascii=True, lower=False, lang=self.clean_lang))
        if view:
            print("Cleaned Tokens: \n{}\n".format(cleaned_tokens))

        # phonemize
        phonemized_tokens = []
        for cleaned_token in cleaned_tokens:
            phonemized_tokens.append(phonemizer.phonemize(cleaned_token, backend="espeak", language=self.g2p_lang,
                                                          preserve_punctuation=True, strip=True, with_stress=True))
        if view:
            print("Phonemes: \n{}\n".format(phonemized_tokens))

        tensors = []
        phones_vector = []
        tags_vector = []
        position_vector = []

        # vectorize and get POS
        if self.use_shallow_pos:
            utt = self.tagger(utt)
        for index, phonemized_token in enumerate(phonemized_tokens):
            for char in phonemized_token:
                phones_vector.append(self.ipa_to_vector.get(char, self.default_vector))
                if self.use_shallow_pos:
                    tags_vector.append(utt[index].pos_)
            phones_vector.append(self.default_vector)
            if self.use_shallow_pos:
                tags_vector.append("SPACE__")

        # generate tensors
        for line in numpy.transpose(numpy.array(phones_vector)):
            tensors.append(torch.tensor(line))
        if self.use_shallow_pos:
            tags_numeric_vector = []
            for el in tags_vector:
                tags_numeric_vector.append(self.tag_id_lookup[el])
            tensors.append(torch.tensor(tags_numeric_vector))
        if self.use_positional_information:
            for index in range(len(phones_vector)):
                position_vector.append(round(index / len(phones_vector), 3))
            tensors.append(torch.tensor(position_vector))
        if view and self.use_shallow_pos:
            tags = []
            for el in utt:
                tags.append(el.tag_)
            print("POS Tags: \n{}\n".format(tags))

        # combine tensors and return
        return torch.stack(tensors, 0)


if __name__ == '__main__':
    tfr = TextFrontend("en")
    print(tfr.string_to_tensor("I own 19,999 cows which I bought for 5.50$!", view=True))
