# pythaiasr

Python Thai ASR

I made a simple python package for Thai ASR. I used model from [chompk/wav2vec2-large-xlsr-thai-tokenized](https://huggingface.co/chompk/wav2vec2-large-xlsr-thai-tokenized).

## Install

```sh
pip install -e .
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