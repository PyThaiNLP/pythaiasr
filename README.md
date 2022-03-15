# PyThaiASR

Python Thai Automatic Speech Recognition

 <a href="https://pypi.python.org/pypi/pythaiasr"><img alt="pypi" src="https://img.shields.io/pypi/v/pythaiasr.svg"/></a><a href="https://opensource.org/licenses/Apache-2.0"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"/></a><a href="https://pepy.tech/project/pythaiasr"><img alt="Download" src="https://pepy.tech/badge/pythaiasr/month"/></a>

PyThaiASR is a Python package for Automatic Speech Recognition with focus on Thai language. It have offline thai automatic speech recognition model from Artificial Intelligence Research Institute of Thailand (AIResearch.in.th).

License: [Apache-2.0 License](https://github.com/PyThaiNLP/pythaiasr/blob/main/LICENSE)

Google Colab: [Link Google colab](https://colab.research.google.com/drive/1zHt3GoxXWCaNSMRzE5lrvpYm9RolcxOW?usp=sharing)

Model homepage: https://huggingface.co/airesearch/wav2vec2-large-xlsr-53-th

## Install

```sh
pip install pythaiasr
```

## Usage

```python
from pythaiasr import asr

file = "a.wav"
print(asr(file))
```
### API

```python
asr(file: str, show_pad: bool = False)
```

- file: path of sound file
- show_pad: show [PAD] in output
- return: thai text from ASR
