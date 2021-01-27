# PreprocesingForTTS

Preprocessing utillity for text and audio to be used in Text-to-Speech. Can transform text into a phoneme ID tensor with prosodic cues, can clean and normalize a wav, can produce an MFCC-bank spectrogram from a wave.

# Installation

Just doing _pip install -r requirements.txt_ should be sufficient, but in case you want to do it manually, here are special cases:

- In the text processing, one module is imported as 'cleantext'. This can however **NOT** be installed using _pip install cleantext_, which gives you a different module. You need to do _pip install clean-text_ (as is in the requirements)

- To install torchaudio using conda you have to specify the host channel, which is pytorch. Download may be very slow due to their hosting.  
_conda install -c pytorch torchaudio_