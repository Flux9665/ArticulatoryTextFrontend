# PreprocesingForTTS

Preprocessing utillity for text and audio to be used in Text-to-Speech. Can transform text into a phoneme ID tensor with prosodic cues, can clean and normalize a wav, can produce an MFCC-bank spectrogram from a wave.

# Installation

Just doing _pip install -r requirements.txt_ should be sufficient, but in case you want to do it manually, here are special cases:

- In the text processing, one module is imported as 'cleantext'. This can however **NOT** be installed using _pip install cleantext_, which gives you a different module. You need to do _pip install clean-text_ (as is in the requirements)

- To install torchaudio using conda you have to specify the host channel, which is pytorch. Download may be very slow due to their hosting.  
_conda install -c pytorch torchaudio_

# Sources 

## Text Processing

The articulation vector lookup is based on PanPhon, it has been modified however to accommodate suprasegmental features and removed some complex phonemes, since the phonemizer used here doesn't produce the complex phonemes.
```
@inproceedings{Mortensen-et-al:2016,
  author    = {David R. Mortensen and
               Patrick Littell and
               Akash Bharadwaj and
               Kartik Goyal and
               Chris Dyer and
               Lori S. Levin},
  title     = {PanPhon: {A} Resource for Mapping {IPA} Segments to Articulatory Feature Vectors},
  booktitle = {Proceedings of {COLING} 2016, the 26th International Conference on Computational Linguistics: Technical Papers},
  pages     = {3475--3484},
  publisher = {{ACL}},
  year      = {2016}
}
```

The phonemizer can be found at https://github.com/bootphon/phonemizer and is using espeak in the backend.

The text cleaner can be found at https://github.com/jfilter/clean-text.

POS tagging for intonation phrase boundary detection is done using Spacy

```
@software{spacy,
  author = {Honnibal, Matthew and Montani, Ines and Van Landeghem, Sofie and Boyd, Adriane},
  title = {{spaCy: Industrial-strength Natural Language Processing in Python}},
  year = 2020,
  publisher = {Zenodo},
  doi = {10.5281/zenodo.1212303},
  url = {https://doi.org/10.5281/zenodo.1212303}
}
```

## Audio Processing

Some of the audio processing is done using Librosa:

```
@inproceedings{mcfee2015librosa,
  title={librosa: Audio and music signal analysis in python},
  author={McFee, Brian and Raffel, Colin and Liang, Dawen and Ellis, Daniel PW and McVicar, Matt and Battenberg, Eric and Nieto, Oriol},
  booktitle={Proceedings of the 14th python in science conference},
  volume={8},
  pages={18--25},
  year={2015},
  organization={Citeseer}
}
```

Audio processing also uses pyloudnorm:

```
Christian Steinmetz, csteinmetz1/pyloudnorm: 0.1.0 (Version v0.1.0), Zenodo, November 2019
```

And a lot of the audio processing is done using the torchaudio module of pytorch:

```
@incollection{NEURIPS2019_9015,
title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
author = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {8024--8035},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf}
}
```

## This Repository

This repository can be cited as follows:

```
@misc{flux2021preprocessing,
  author = {Florian Lux},
  title = {PreprocessingForTTS},
  year = {2021},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Flux9665/PreprocessingForTTS}}
}
```
