# PyThaiASR

Python Thai Automatic Speech Recognition

 <a href="https://pypi.python.org/pypi/pythaiasr"><img alt="pypi" src="https://img.shields.io/pypi/v/pythaiasr.svg"/></a><a href="https://opensource.org/licenses/Apache-2.0"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"/></a><a href="https://pepy.tech/project/pythaiasr"><img alt="Download" src="https://pepy.tech/badge/pythaiasr/month"/></a>[![Coverage Status](https://coveralls.io/repos/github/PyThaiNLP/pythaiasr/badge.svg)](https://coveralls.io/github/PyThaiNLP/pythaiasr)

PyThaiASR is a Python package for Automatic Speech Recognition with focus on Thai language. It have offline thai automatic speech recognition model.

License: [Apache-2.0 License](https://github.com/PyThaiNLP/pythaiasr/blob/main/LICENSE)

Google Colab: [Link Google colab](https://colab.research.google.com/github/PyThaiNLP/pythaiasr/blob/main/examples/pythaiasr-colab.ipynb)

## Install

```sh
pip install pythaiasr
```

By default, PyThaiASR uses **Typhoon ASR** powered by ONNX Runtime for low-latency, lightweight offline and realtime speech recognition on CPU and GPU. On first use, Typhoon model files are automatically downloaded to `~/pythaiasr-data/typhoon-asr-realtime/`.

**For PyTorch & Transformers models (Wav2Vec2 / Whisper):**
If you want to use the Wav2Vec2 or Whisper models:

```sh
pip install pythaiasr[torch]
```

**For Wav2Vec2 with language model:**
If you want to use `wannaphong/wav2vec2-large-xlsr-53-th-cv8-*` with a language model:

```sh
pip install pythaiasr[lm]
pip install https://github.com/kpu/kenlm/archive/refs/heads/master.zip
```

**For live audio streaming:**
If you want to stream live audio from your microphone:

```sh
pip install pythaiasr[stream]
```

## Usage

### File-based ASR

```python
from pythaiasr import asr

file = "sample.wav"

# Uses Typhoon ASR (FastConformer RNN-T ONNX) by default
print(asr(file))

# Or explicitly select another model (requires pythaiasr[torch])
# print(asr(file, model="airesearch/wav2vec2-large-xlsr-53-th"))
# print(asr(file, model="biodatlab/whisper-small-th-combined"))
```

### Live Audio Streaming

Stream audio directly from your microphone/soundcard in real-time:

```python
from pythaiasr import stream_asr

# Streams audio in real-time using Typhoon ASR by default
for transcription in stream_asr():
    print(transcription, end=" ", flush=True)
    # Press Ctrl+C to stop
```

And examples/stream_example.py

### Real-Time Streaming from File or Microphone

```python
from pythaiasr import FastConformerRNNT, RealtimeStreamASR, stream_from_file, stream_from_mic

model = FastConformerRNNT(device="auto")
streamer = RealtimeStreamASR(model=model, step_sec=0.48)

# Simulate streaming from a pre-recorded audio file
stream_from_file(streamer, "sample.wav")

# Or stream live from microphone with sounddevice
# stream_from_mic(streamer)
```

### API

#### asr

```python
asr(data: str, model: str = _model_name, lm: bool=False, device: str=None, sampling_rate: int=16_000)
```

- data: path of sound file or numpy array of the voice
- model: The ASR model (default: `typhoon_asr`)
- lm: Use language model (for wav2vec2 models with LM)
- device: device (`auto`, `cpu`, `cuda`)
- sampling_rate: The sample rate
- return: thai text from ASR

#### stream_asr

```python
stream_asr(model: str = _model_name, lm: bool=False, device: str=None, chunk_duration: float=None, sampling_rate: int=16_000)
```

- model: The ASR model (default: `typhoon_asr`)
- lm: Use language model (for wav2vec2 models with LM)
- device: device for running model
- chunk_duration: Duration of each audio chunk in seconds (default: 0.48s for Typhoon, 5.0s for others)
- sampling_rate: The sample rate (default: 16000)
- yield: Thai text transcription from each audio chunk

**Options for model**
- *typhoon_asr* / *typhoon-asr-realtime* (default) - Typhoon FastConformer RNN-T ONNX model (offline & realtime)
- *airesearch/wav2vec2-large-xlsr-53-th* - AI RESEARCH - PyThaiNLP model (requires pythaiasr[torch])
- *wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm* - Thai Wav2Vec2 with CommonVoice V8 (newmm tokenizer) (requires pythaiasr[torch])
- *wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut* - Thai Wav2Vec2 with CommonVoice V8 (deepcut tokenizer) (requires pythaiasr[torch])
- *biodatlab/whisper-small-th-combined* - Thai Whisper small model (requires pythaiasr[torch])
- *biodatlab/whisper-th-medium-combined* - Thai Whisper medium model (requires pythaiasr[torch])
- *biodatlab/whisper-th-large-combined* - Thai Whisper large model (requires pythaiasr[torch])

You can read about models from the list:

- [*typhoon-ai/typhoon-asr-realtime* / *wannaphong/typhoon-asr-realtime-onnx* - Typhoon FastConformer RNN-T ONNX model](https://huggingface.co/wannaphong/typhoon-asr-realtime-onnx)
- [*airesearch/wav2vec2-large-xlsr-53-th* - AI RESEARCH - PyThaiNLP model](https://medium.com/airesearch-in-th/airesearch-in-th-3c1019a99cd)
- [*annaphong/wav2vec2-large-xlsr-53-th-cv8-newmm* - Thai Wav2Vec2 with CommonVoice V8 (newmm tokenizer) + language model](https://huggingface.co/wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm) 
- [*wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut* - Thai Wav2Vec2 with CommonVoice V8 (deepcut tokenizer) + language model](https://huggingface.co/wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut)
- [*biodatlab/whisper-small-th-combined* - Thai Whisper small model](https://huggingface.co/biodatlab/whisper-small-th-combined)
- [*biodatlab/whisper-th-medium-combined* - Thai Whisper medium model](https://huggingface.co/biodatlab/whisper-th-medium-combined)
- [*biodatlab/whisper-th-large-combined* - Thai Whisper large model](https://huggingface.co/biodatlab/whisper-th-large-combined)

### Docker
To use this inside of Docker do the following:
```sh
docker build -t <Your Tag name> .
docker run docker run --entrypoint /bin/bash -it <Your Tag name>
```
You will then get access to a interactive shell environment where you can use python with all packages installed.
