import sys

import phonemizer
import spacy
import torch
from cleantext import clean


class TextFrontend:
    def __init__(self, language):
        """
        Mostly loading the right spacy
        models and preparing ID lookups
        """
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

        # SETUP POS TAGGER AND LOOKUP
        self.tagger = self.nlp.get_pipe("tagger")
        content_word_tags = ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"]
        function_word_tags = ["ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"]
        other_tags = ["PUNCT", "SYM", "X"]
        self.tag_id_lookup = {"SPACE__": 0}
        self.dep_id_lookup = {"SPACE__": 0}
        for tag in content_word_tags:
            self.tag_id_lookup[tag] = 1
        for tag in function_word_tags:
            self.tag_id_lookup[tag] = 2
        for tag in other_tags:
            self.tag_id_lookup[tag] = 3

    def string_to_tensor(self, text, include_prosodic_cues=True, view=False):
        """
        applies the entire pipeline to a text
        """
        utt = self.tokenizer(text)

        cleaned_tokens = []
        for token in utt:
            cleaned_tokens.append(clean(token, fix_unicode=True, to_ascii=True, lower=False, lang=self.clean_lang))
        if view:
            print(cleaned_tokens)

        phonemized_tokens = []
        for cleaned_token in cleaned_tokens:
            phonemized_tokens.append(phonemizer.phonemize(cleaned_token, backend="espeak", language=self.g2p_lang,
                                                          preserve_punctuation=True, strip=True, with_stress=True))

        if include_prosodic_cues:
            phones_vector = []
            tags_vector = []
            utt = self.tagger(utt)
            for index, phonemized_token in enumerate(phonemized_tokens):
                for char in phonemized_token:
                    phones_vector.append(ord(char))
                    tags_vector.append(utt[index].pos_)
                phones_vector.append(ord(' '))
                tags_vector.append("SPACE__")

            tags_numeric_vector = []
            for el in tags_vector:
                tags_numeric_vector.append(self.tag_id_lookup[el])

            phones_tensor = torch.tensor(phones_vector)
            tags_tensor = torch.tensor(tags_numeric_vector)
            rich_tensor = torch.stack((phones_tensor, tags_tensor), 0)
            if view:
                tags = []
                for el in utt:
                    tags.append(el.tag_)
                print(phonemized_tokens)
                print(tags)

            return rich_tensor

        else:
            vector = []
            for index, phonemized_token in enumerate(phonemized_tokens):
                vector.append(ord(' '))
                for char in phonemized_token:
                    vector.append(ord(char))
            tensor = torch.tensor(vector)
            if view:
                print(phonemized_tokens)
            return tensor


if __name__ == '__main__':
    tfr_1 = TextFrontend("en")
    print(tfr_1.string_to_tensor("I own 19,999 cows which I bought for 5.50$!", view=True))
