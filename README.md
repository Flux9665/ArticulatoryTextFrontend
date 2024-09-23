## Graphemes to Articulatory Features in over 7000 Languages

This is a text-processing frontend that converts graphemes to phonemes and then further converts those phonemes into
articulatory features, for over 7000 languages.

Have a look at `run_grapheme_to_feature_demo.py` for a demonstration on how to use this tool.

Also, check out the interactive demo to get an impression without any installation
hassle: https://huggingface.co/spaces/Flux9665/ArticulatoryFeatures

This is a standalone version of the text frontend used in the IMS Toucan TTS
toolkit: https://github.com/DigitalPhonetics/IMS-Toucan

<br><br>

If you find this useful, consider giving the repository a star. And if you use this in a scientific work, here are a few
bibtex entries for your convenience:

initial version including all IPA phonemes, but ignoring any IPA modifiers:

```
@inproceedings{lux2022laml,
  year         = 2022,
  title        = {{Language-Agnostic Meta-Learning for Low-Resource Text-to-Speech with Articulatory Features}},
  author       = {Florian Lux and Ngoc Thang Vu},
  booktitle    = {ACL}
}
```

updated version that includes all IPA phonemes, as well as the IPA modifiers:

```
@inproceedings{lux2022lrms,
  year         = 2022,
  title        = {{Low-Resource Multilingual and Zero-Shot Multispeaker TTS}},
  author       = {Florian Lux and Julia Koch and Ngoc Thang Vu},
  booktitle    = {AACL}
}
```

updated version extending this to over 7000 languages:

```
@inproceedings{lux2024massive,
  year         = 2024,
  title        = {{Meta Learning Text-to-Speech Synthesis in over 7000 Languages}},
  author       = {Florian Lux and Sarina Meyer and Lyonel Behringer and Frank Zalkow and Phat Do and Matt Coler and  Emanuël A. P. Habets and Ngoc Thang Vu},
  booktitle    = {Interspeech}
  publisher    = {ISCA}
}
```