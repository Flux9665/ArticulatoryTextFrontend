import sys

import phonemizer
import spacy
import torch
from cleantext import clean
from num2words import num2words
from polyglot.text import Word


class TextFrontend:
    def __init__(self, language, use_shallow_pos=True, use_morphology=True):
        """
        Mostly loading the right spacy
        models and preparing ID lookups
        """
        self.use_shallow_pos = use_shallow_pos
        self.use_morphology = use_morphology

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
            self.dep_id_lookup = {"SPACE__": 0}
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
        utt = self.tokenizer(text)

        cleaned_tokens = []
        for index, token in enumerate(utt):
            if str(token).replace(".", "").replace(",", "").isnumeric():
                to = "cardinal"
                if self.clean_lang == "en":
                    num = num2words(float(str(token).replace(",", "")), lang=self.clean_lang, to=to)
                elif self.clean_lang == "de":
                    num = num2words(float(str(token).replace(".", "")), lang=self.clean_lang, to=to)
                if len(utt) > index + 1:
                    if str(utt[index + 1]) == "$" or str(utt[index + 1]) == "€":
                        num = ""
                        if self.clean_lang == "en":
                            for el in str(token).split("."):
                                num += num2words(float(el), lang=self.clean_lang, to=to) + " "
                        elif self.clean_lang == "de":
                            for el in str(token).split(","):
                                num += num2words(float(el), lang=self.clean_lang, to=to) + " "
                elif len(str(token)) == 4 and str(token).isnumeric():
                    to = "year"
                cleaned_tokens.append(clean(num.rstrip(" "),
                                            fix_unicode=True,
                                            to_ascii=True, lower=False,
                                            lang=self.clean_lang))
            else:
                cleaned_tokens.append(clean(token, fix_unicode=True, to_ascii=True, lower=False, lang=self.clean_lang))
        if view:
            print("Cleaned Tokens: \n{}\n".format(cleaned_tokens))

        phonemized_tokens = []
        if view:
            morphemes = []
        for cleaned_token in cleaned_tokens:
            if self.use_morphology:
                phonemized_token = ""
                for morpheme in Word(cleaned_token, language=self.clean_lang).morphemes:
                    phonemized_token += phonemizer.phonemize(morpheme, backend="espeak", language=self.g2p_lang,
                                                             preserve_punctuation=True, strip=True,
                                                             with_stress=True)[0]
                    phonemized_token += "|"
                    if view:
                        morphemes.append(morpheme)
                phonemized_tokens.append(phonemized_token.rstrip("|"))

            else:
                phonemized_tokens.append(phonemizer.phonemize(cleaned_token, backend="espeak", language=self.g2p_lang,
                                                              preserve_punctuation=True, strip=True, with_stress=True))

        if view:
            print("Morphemes: \n{}\n".format(morphemes))

        tensors = []
        phones_vector = []
        tags_vector = []

        if self.use_shallow_pos:
            utt = self.tagger(utt)
        for index, phonemized_token in enumerate(phonemized_tokens):
            for char in phonemized_token:
                phones_vector.append(ord(char))
                if self.use_shallow_pos:
                    tags_vector.append(utt[index].pos_)
            phones_vector.append(ord(' '))
            if self.use_shallow_pos:
                tags_vector.append("SPACE__")
        tensors.append(torch.tensor(phones_vector))
        if self.use_shallow_pos:
            tags_numeric_vector = []
            for el in tags_vector:
                tags_numeric_vector.append(self.tag_id_lookup[el])
            tensors.append(torch.tensor(tags_numeric_vector))

        if view:
            print("Phonemes (optionally with morpheme boundaries): \n{}\n".format(phonemized_tokens))
            if self.use_shallow_pos:
                tags = []
                for el in utt:
                    tags.append(el.tag_)
                print("POS Tags (internally simplified to content, function, other): \n{}\n".format(tags))

        return torch.stack(tensors, 0)


if __name__ == '__main__':
    tfr = TextFrontend("en")
    print(tfr.string_to_tensor("I own 19,999 cows which I bought for 5.50$!", view=True))
