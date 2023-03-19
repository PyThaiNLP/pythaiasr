# PyThaiASR

Python Thai Automatic Speech Recognition

 <a href="https://pypi.python.org/pypi/pythaiasr"><img alt="pypi" src="https://img.shields.io/pypi/v/pythaiasr.svg"/></a><a href="https://opensource.org/licenses/Apache-2.0"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"/></a><a href="https://pepy.tech/project/pythaiasr"><img alt="Download" src="https://pepy.tech/badge/pythaiasr/month"/></a>[![Coverage Status](https://coveralls.io/repos/github/PyThaiNLP/pythaiasr/badge.svg)](https://coveralls.io/github/PyThaiNLP/pythaiasr)

PyThaiASR is a Python package for Automatic Speech Recognition with focus on Thai language. It have offline thai automatic speech recognition model.

License: [Apache-2.0 License](https://github.com/PyThaiNLP/pythaiasr/blob/main/LICENSE)

Google Colab: [Link Google colab](https://colab.research.google.com/drive/1zHt3GoxXWCaNSMRzE5lrvpYm9RolcxOW?usp=sharing)

Model homepage: https://huggingface.co/airesearch/wav2vec2-large-xlsr-53-th

## Install

```sh
pip install pythaiasr
```

**For Wav2Vec2 with language model:**
if you want to use wannaphong/wav2vec2-large-xlsr-53-th-cv8-* model with language model, you needs to install by the step.

```sh
pip install pythaiasr[lm]
pip install https://github.com/kpu/kenlm/archive/refs/heads/master.zip
```

## Usage

```python
from pythaiasr import asr

file = "a.wav"
print(asr(file))
```
### API

```python
asr(data: str, model: str = _model_name, lm: bool=False, device: str=None, sampling_rate: int=16_000)
```

- data: path of sound file or numpy array of the voice
- model: The ASR model
- lm: Use language model (except *airesearch/wav2vec2-large-xlsr-53-th* model)
- device: device
- sampling_rate: The sample rate
- return: thai text from ASR

**Options for model**
- *airesearch/wav2vec2-large-xlsr-53-th* (default) - AI RESEARCH - PyThaiNLP model
- *wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm* - Thai Wav2Vec2 with CommonVoice V8 (newmm tokenizer)
- *wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut* - Thai Wav2Vec2 with CommonVoice V8 (deepcut tokenizer)

You can read about models from the list:

- [*airesearch/wav2vec2-large-xlsr-53-th* - AI RESEARCH - PyThaiNLP model](https://medium.com/airesearch-in-th/airesearch-in-th-3c1019a99cd)
- [*annaphong/wav2vec2-large-xlsr-53-th-cv8-newmm* - Thai Wav2Vec2 with CommonVoice V8 (newmm tokenizer) + language model](https://huggingface.co/wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm) 
- [*wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut* - Thai Wav2Vec2 with CommonVoice V8 (deepcut tokenizer) + language model](https://huggingface.co/wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut)

### Docker
To use this inside of Docker do the following:
```sh
docker build -t <Your Tag name> .
docker run docker run --entrypoint /bin/bash -it <Your Tag name>
```
You will then get access to a interactive shell environment where you can use python with all packages installed.
