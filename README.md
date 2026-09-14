# PyThaiASR

Python Thai Automatic Speech Recognition

 <a href="https://pypi.python.org/pypi/pythaiasr"><img alt="pypi" src="https://img.shields.io/pypi/v/pythaiasr.svg"/></a><a href="https://opensource.org/licenses/Apache-2.0"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"/></a><a href="https://pepy.tech/project/pythaiasr"><img alt="Download" src="https://pepy.tech/badge/pythaiasr/month"/></a>[![Coverage Status](https://coveralls.io/repos/github/PyThaiNLP/pythaiasr/badge.svg)](https://coveralls.io/github/PyThaiNLP/pythaiasr)

PyThaiASR is a Python package for Automatic Speech Recognition with focus on Thai language. It have offline thai automatic speech recognition model.

License: [Apache-2.0 License](https://github.com/PyThaiNLP/pythaiasr/blob/main/LICENSE)

Google Colab: [Link Google colab](https://colab.research.google.com/github/PyThaiNLP/pythaiasr/blob/main/examples/pythaiasr_timestamps_diarize.ipynb)

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

**For Sherpa-ONNX Diarization Backend (Optional):**
PyThaiASR includes a built-in ONNX diarization engine out-of-the-box requiring no extra dependencies. If you prefer using the optional Sherpa-ONNX backend:

```sh
pip install pythaiasr[diarize]
```

## Usage

### File-based ASR

```python
from pythaiasr import asr

file = "sample.wav"

# Uses Typhoon ASR (FastConformer RNN-T ONNX) by default
print(asr(file))

# With timestamps (returns dictionary with 'text', 'chunks', and 'timestamps')
result = asr(file, return_timestamps=True)
print(result["text"])
for chunk in result["chunks"]:
    print(f"[{chunk['start']:.2f}s -> {chunk['end']:.2f}s] {chunk['text']}")

# Or explicitly select another model (requires pythaiasr[torch])
# print(asr(file, model="airesearch/wav2vec2-large-xlsr-53-th"))
# print(asr(file, model="biodatlab/whisper-small-th-combined"))
# print(asr(file, model="biodatlab/whisper-th-medium-timestamp", return_timestamps=True))
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

### Speech Diarization (Who Spoke When)

Detect speaker turns and timestamps using ONNX:

```python
from pythaiasr import diarize

# Identify who spoke when
segments = diarize("meeting.wav")
for seg in segments:
    print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}")
```

### Speech Diarization + ASR (`asr_diarize`)

Detect speakers and transcribe each speaker turn with ASR:

```python
from pythaiasr import asr_diarize

# Attributed transcription per speaker
turns = asr_diarize("meeting.wav", asr_model="typhoon_asr")
for turn in turns:
    print(f"[{turn['start']:.2f}s - {turn['end']:.2f}s] {turn['speaker']}: {turn['text']}")
```

See examples/diarize_example.py

```
============================================================
1. Speech Diarization (Who Spoke When)
============================================================
Processing: examples/../tests/test-diarize.wav ...
[  0.17s ->   1.87s] SPEAKER_02
[  1.97s ->   4.30s] SPEAKER_01
[  4.83s ->   6.49s] SPEAKER_02
[  6.78s ->   8.36s] SPEAKER_01

============================================================
2. Diarization + Speech Recognition (ASR Diarize)
============================================================
Transcribing turns with Typhoon ASR: examples/../tests/test-diarize.wav ...
[  0.17s ->   1.87s] SPEAKER_02: สวัสดีชาวโลกทุกท่าน
[  1.97s ->   4.30s] SPEAKER_01: แล้วระบบนี้ทํางานอย่างไร
[  4.83s ->   6.49s] SPEAKER_02: ใช้ปัญญาประดิษฐ์ในการทดสอบ
[  6.78s ->   8.36s] SPEAKER_01: ใช้งานได้ดีทีเดียว
```

### API

#### asr

```python
asr(
    data: Union[str, np.ndarray],
    model: str = _model_name,
    lm: bool = False,
    device: str = None,
    sampling_rate: int = 16_000,
    return_timestamps: Optional[Union[bool, str]] = None,
    timestamps: Optional[Union[bool, str]] = None,
)
```

- data: path of sound file or numpy array of the voice
- model: The ASR model (default: `typhoon_asr`)
- lm: Use language model (for wav2vec2 models with LM)
- device: device (`auto`, `cpu`, `cuda`)
- sampling_rate: The sample rate
- return_timestamps: Return timestamps dictionary (`True`, `"word"`, or `"char"`)
- timestamps: Alias for `return_timestamps`
- return: Thai text from ASR (`str`) or dictionary with `"text"`, `"chunks"`, and `"timestamps"` if `return_timestamps=True`

#### stream_asr

```python
stream_asr(
    model: str = _model_name,
    lm: bool = False,
    device: str = None,
    chunk_duration: float = None,
    sampling_rate: int = 16_000,
    return_timestamps: bool = False,
    timestamps: Optional[bool] = None,
)
```

- model: The ASR model (default: `typhoon_asr`)
- lm: Use language model (for wav2vec2 models with LM)
- device: device for running model
- chunk_duration: Duration of each audio chunk in seconds (default: 0.48s for Typhoon, 5.0s for others)
- sampling_rate: The sample rate (default: 16000)
- return_timestamps: Yield dictionary with text and chunk timestamps (`True` or `False`)
- timestamps: Alias for `return_timestamps`
- yield: Thai text transcription (or dict with timestamp) from each audio chunk

**Options for model**
- *typhoon_asr* / *typhoon-asr-realtime* (default) - Typhoon FastConformer RNN-T ONNX model (offline & realtime)
- *airesearch/wav2vec2-large-xlsr-53-th* - AI RESEARCH - PyThaiNLP model (requires pythaiasr[torch])
- *wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm* - Thai Wav2Vec2 with CommonVoice V8 (newmm tokenizer) (requires pythaiasr[torch])
- *wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut* - Thai Wav2Vec2 with CommonVoice V8 (deepcut tokenizer) (requires pythaiasr[torch])
- *biodatlab/whisper-small-th-combined* - Thai Whisper small model (requires pythaiasr[torch])
- *biodatlab/whisper-th-medium-combined* - Thai Whisper medium model (requires pythaiasr[torch])
- *biodatlab/whisper-th-large-combined* - Thai Whisper large model (requires pythaiasr[torch])
- *biodatlab/whisper-th-medium-timestamp* - Thai Whisper medium model with timestamp support (requires pythaiasr[torch])

You can read about models from the list:

- [*typhoon-ai/typhoon-asr-realtime* / *wannaphong/typhoon-asr-realtime-onnx* - Typhoon FastConformer RNN-T ONNX model](https://huggingface.co/wannaphong/typhoon-asr-realtime-onnx)
- [*airesearch/wav2vec2-large-xlsr-53-th* - AI RESEARCH - PyThaiNLP model](https://medium.com/airesearch-in-th/airesearch-in-th-3c1019a99cd)
- [*annaphong/wav2vec2-large-xlsr-53-th-cv8-newmm* - Thai Wav2Vec2 with CommonVoice V8 (newmm tokenizer) + language model](https://huggingface.co/wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm) 
- [*wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut* - Thai Wav2Vec2 with CommonVoice V8 (deepcut tokenizer) + language model](https://huggingface.co/wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut)
- [*biodatlab/whisper-small-th-combined* - Thai Whisper small model](https://huggingface.co/biodatlab/whisper-small-th-combined)
- [*biodatlab/whisper-th-medium-combined* - Thai Whisper medium model](https://huggingface.co/biodatlab/whisper-th-medium-combined)
- [*biodatlab/whisper-th-large-combined* - Thai Whisper large model](https://huggingface.co/biodatlab/whisper-th-large-combined)
- [*biodatlab/whisper-th-medium-timestamp* - Thai Whisper medium model with timestamp support](https://huggingface.co/biodatlab/whisper-th-medium-timestamp)

#### diarize

```python
diarize(
    data: Union[str, Path, np.ndarray],
    model: str = "pyannote_segmentation",
    device: Optional[str] = None,
    sampling_rate: int = 16_000,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    onset: float = 0.5,
    offset: float = 0.5,
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.5,
    backend: str = "onnx",
) -> List[Dict[str, Union[float, str]]]
```

- `data`: Audio file path or 1D numpy array of audio waveform.
- `model`: Diarization model identifier (default: `"pyannote_segmentation"`).
- `device`: Device to run inference on (`"auto"`, `"cpu"`, `"cuda"`).
- `sampling_rate`: Audio sampling rate (default: 16000).
- `num_speakers`: Exact number of speakers if known.
- `onset`: Speech onset probability threshold (default: 0.5).
- `offset`: Speech offset probability threshold (default: 0.5).
- `min_duration_on`: Minimum speaker turn duration in seconds (default: 0.3).
- `min_duration_off`: Minimum silence duration to split turns in seconds (default: 0.5).
- `backend`: Diarization engine (`"onnx"` or `"sherpa-onnx"`, default: `"onnx"`).
- **Returns**: List of segments with `start`, `end`, and `speaker` keys.

#### asr_diarize

```python
asr_diarize(
    data: Union[str, Path, np.ndarray],
    asr_model: str = "typhoon_asr",
    diarize_model: str = "pyannote_segmentation",
    device: Optional[str] = None,
    sampling_rate: int = 16_000,
    lm: bool = False,
    num_speakers: Optional[int] = None,
    merge_same_speaker: bool = True,
    max_merge_gap: float = 0.5,
    backend: str = "onnx",
    **kwargs,
) -> List[Dict[str, Union[float, str]]]
```

- `data`: Audio file path or 1D numpy array of audio waveform.
- `asr_model`: The ASR model name (default: `"typhoon_asr"`).
- `diarize_model`: Diarization model name (default: `"pyannote_segmentation"`).
- `merge_same_speaker`: Whether to merge adjacent speech turns from the same speaker (default: `True`).
- `max_merge_gap`: Maximum gap in seconds between same-speaker segments to merge (default: 0.5).
- `backend`: Diarization engine (`"onnx"` or `"sherpa-onnx"`, default: `"onnx"`).
- **Returns**: List of speaker turns with `start`, `end`, `speaker`, and `text` keys.

### Docker
To use this inside of Docker do the following:
```sh
docker build -t <Your Tag name> .
docker run docker run --entrypoint /bin/bash -it <Your Tag name>
```
You will then get access to a interactive shell environment where you can use python with all packages installed.
