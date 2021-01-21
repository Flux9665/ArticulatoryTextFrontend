# PreprocesingForTTS

Preprocessing utillity for text and audio to be used in Text-to-Speech. Can transform text into a phoneme ID tensor with prosodic cues, can clean and normalize a wav, can produce an MFCC spectrogram from a wave.

# Installation

Some things are not so straightforward, so they are listed here:

- In the text processing, one module is imported as 'cleantext'. This can however **NOT** be installed using _pip install cleantext_, which gives you a different module. You need to do _pip install clean-text_ (as is in the requirements)

- To use the spacy models for prosodic cue extraction in the text processing, you need to download the German and English models:  
_python -m spacy download en_core_web_sm_  
_python -m spacy download de_core_news_sm_

- To install torchaudio you have to specify the host channel, which is pytorch. Download may be very slow due to their hosting.  
_conda install -c pytorch torchaudio_
