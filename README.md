# pythaiasr

Python Thai ASR

I made a simple python package for Thai ASR. I used model from [chompk/wav2vec2-large-xlsr-thai-tokenized](https://huggingface.co/chompk/wav2vec2-large-xlsr-thai-tokenized).

## Install

```sh
pip install -e .
```

## Usege

```python
from pythaiasr import asr

file = "a.wav"
print(asr(file))
```