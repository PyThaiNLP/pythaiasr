# -*- coding: utf-8 -*-
import os
import sys
import logging
from typing import Optional, Union

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import torchaudio
except ImportError:
    torch = None
    torchaudio = None

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity(40)
except ImportError:
    pass

from pythaiasr.download import (
    get_pythaiasr_path,
    get_typhoon_model_files,
    get_diarization_model_files,
    download_file,
)
from pythaiasr.typhoon import (
    FastConformerRNNT,
    RealtimeStreamASR,
    StreamingTranscriber,
    stream_from_mic,
    stream_from_file,
    list_audio_devices,
    extract_features,
    load_audio,
)
from pythaiasr.diarization import (
    Diarization,
    diarize,
    asr_diarize,
    merge_same_speaker_segments,
)

# Friendly alias
TyphoonASR = FastConformerRNNT


class ASR:
    def __init__(self, model: str="typhoon_asr", lm: bool=False, device: str=None) -> None:
        """
        :param str model: The ASR model name
        :param bool lm: Use language model (default is False and except *airesearch/wav2vec2-large-xlsr-53-th* model)
        :param str device: device
        
        **Options for model**
            * *typhoon_asr* / *typhoon-asr-realtime* (default) - Typhoon FastConformer RNN-T ONNX model (offline & realtime)
            * *airesearch/wav2vec2-large-xlsr-53-th* - AI RESEARCH - PyThaiNLP model (requires pythaiasr[torch])
            * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm* - Thai Wav2Vec2 with CommonVoice V8 (newmm tokenizer) + language model (requires pythaiasr[torch])
            * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut* - Thai Wav2Vec2 with CommonVoice V8 (deepcut tokenizer) + language model (requires pythaiasr[torch])
            * *biodatlab/whisper-small-th-combined* - Thai Whisper small model (requires pythaiasr[torch])
            * *biodatlab/whisper-th-medium-combined* - Thai Whisper medium model (requires pythaiasr[torch])
            * *biodatlab/whisper-th-large-combined* - Thai Whisper large model (requires pythaiasr[torch])
            * *biodatlab/whisper-th-medium-timestamp* - Thai Whisper medium model with timestamp support (requires pythaiasr[torch])
        """
        self.model_name = model
        self.support_model = [
            "airesearch/wav2vec2-large-xlsr-53-th",
            "wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm",
            "wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut",
            "biodatlab/whisper-small-th-combined",
            "biodatlab/whisper-th-medium-combined",
            "biodatlab/whisper-th-large-combined",
            "biodatlab/whisper-th-medium-timestamp",
            "typhoon_asr",
            "typhoon-asr-realtime",
            "wannaphong/typhoon-asr-realtime-onnx",
        ]
        self.whisper_models = [
            "biodatlab/whisper-small-th-combined",
            "biodatlab/whisper-th-medium-combined",
            "biodatlab/whisper-th-large-combined",
            "biodatlab/whisper-th-medium-timestamp",
        ]
        self.typhoon_models = [
            "typhoon_asr",
            "typhoon-asr-realtime",
            "wannaphong/typhoon-asr-realtime-onnx",
        ]
        assert self.model_name in self.support_model, f"Model {self.model_name} is not in supported models: {self.support_model}"
        self.lm = lm

        self.is_typhoon = self.model_name in self.typhoon_models
        self.is_whisper = self.model_name in self.whisper_models

        if self.is_typhoon:
            dev = device if device is not None else "auto"
            self.model = FastConformerRNNT(device=dev)
            self.device = dev
            return

        if torch is None or torchaudio is None:
            raise ImportError(
                f"torch and torchaudio are required for {self.model_name}. "
                "Install them with: pip install pythaiasr[torch]"
            )

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.is_whisper:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        elif not self.lm:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
        else:
            from transformers import AutoProcessor, AutoModelForCTC
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForCTC.from_pretrained(self.model_name).to(self.device)

    def speech_file_to_array_fn(self, batch: dict) -> dict:
        if torchaudio is None:
            raise ImportError("torchaudio is required for audio loading. Install it with: pip install torchaudio")
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        batch["speech"] = speech_array[0]
        batch["sampling_rate"] = sampling_rate
        return batch

    def resample(self, batch: dict) -> dict:
        if torchaudio is None:
            raise ImportError("torchaudio is required for audio resampling. Install it with: pip install torchaudio")
        resampler = torchaudio.transforms.Resample(batch['sampling_rate'], 16_000)
        batch["speech"] = resampler(batch["speech"]).numpy()
        batch["sampling_rate"] = 16_000
        return batch

    def prepare_dataset(self, batch: dict) -> dict:
        batch["input_values"] = self.processor(batch["speech"], sampling_rate=batch["sampling_rate"]).input_values
        return batch
    
    def __call__(
        self,
        data: Union[str, np.ndarray],
        sampling_rate: int = 16_000,
        return_timestamps: Optional[Union[bool, str]] = None,
        timestamps: Optional[Union[bool, str]] = None,
    ) -> Union[str, dict]:
        """
        :param Union[str, np.ndarray] data: path of sound file or numpy array of the voice
        :param int sampling_rate: The sample rate
        :param Optional[Union[bool, str]] return_timestamps: If True, returns a dictionary with
            text, chunks, and timestamps. Can also be "word" or "char".
        :param Optional[Union[bool, str]] timestamps: Alias for return_timestamps.
        :return: Thai text string from ASR or dictionary with timestamps.
        :rtype: Union[str, dict]
        """
        use_timestamps = return_timestamps if return_timestamps is not None else timestamps
        if self.is_typhoon:
            return self.model.transcribe(
                data,
                sample_rate=sampling_rate,
                return_timestamps=use_timestamps,
            )

        b = {}
        if isinstance(data, np.ndarray):
            b["speech"] = data
            b["sampling_rate"] = sampling_rate
            _preprocessing = b
        else:
            b["path"] = data
            _preprocessing = self.speech_file_to_array_fn(b)
        
        if self.is_whisper:
            # Whisper model processing
            if b["sampling_rate"] != 16_000:
                speech_tensor = b["speech"] if isinstance(b["speech"], torch.Tensor) else torch.tensor(b["speech"])
                resampler = torchaudio.transforms.Resample(b['sampling_rate'], 16_000)
                b["speech"] = resampler(speech_tensor).numpy()
                b["sampling_rate"] = 16_000
            
            input_features = self.processor(b["speech"], sampling_rate=16_000, return_tensors="pt").input_features
            input_features = input_features.to(self.device)
            
            if use_timestamps:
                predicted_ids = self.model.generate(input_features, return_timestamps=True)
                pred_cpu = predicted_ids[0].cpu() if hasattr(predicted_ids[0], "cpu") else predicted_ids[0]
                chunks = []
                txt = ""
                try:
                    decoded = self.processor.tokenizer.decode(
                        pred_cpu,
                        skip_special_tokens=True,
                        output_offsets=True,
                    )
                    txt = decoded.get("text", "").strip()
                    raw_offsets = decoded.get("offsets", [])
                    for off in raw_offsets:
                        ts = off.get("timestamp", (0.0, 0.0))
                        c_text = off.get("text", "").strip()
                        s_time = round(float(ts[0]), 2) if ts[0] is not None else 0.0
                        e_time = round(float(ts[1]), 2) if ts[1] is not None else s_time
                        chunks.append({
                            "text": c_text,
                            "timestamp": (s_time, e_time),
                            "start": s_time,
                            "end": e_time,
                        })
                except Exception:
                    # Fallback decoding with timestamp tokens parsing
                    raw_text = self.processor.tokenizer.decode(pred_cpu, skip_special_tokens=False)
                    txt = self.processor.tokenizer.decode(pred_cpu, skip_special_tokens=True).strip()
                    import re
                    matches = list(re.finditer(r"<\|([0-9\.]+)\|>", raw_text))
                    for i in range(len(matches) - 1):
                        s_time = round(float(matches[i].group(1)), 2)
                        e_time = round(float(matches[i + 1].group(1)), 2)
                        seg_text = raw_text[matches[i].end():matches[i + 1].start()].strip()
                        if seg_text:
                            chunks.append({
                                "text": seg_text,
                                "timestamp": (s_time, e_time),
                                "start": s_time,
                                "end": e_time,
                            })
                return {
                    "text": txt,
                    "chunks": chunks,
                    "timestamps": chunks,
                }
            else:
                predicted_ids = self.model.generate(input_features)
                txt = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                return txt
        else:
            # Wav2Vec2 model processing
            a = self.prepare_dataset(b)
            input_dict = self.processor(a["input_values"][0], return_tensors="pt", padding=True).to(self.device)
            logits = self.model(input_dict.input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)[0]

            if use_timestamps:
                ratio = getattr(self.model.config, "inputs_to_logits_ratio", 320)
                sr = b.get("sampling_rate", 16000)
                time_offset = ratio / float(sr)
                chunks = []
                offsets = []

                if self.lm:
                    try:
                        decoded = self.processor.decode(
                            logits.detach().cpu().numpy()[0],
                            output_word_offsets=True,
                        )
                        txt = getattr(decoded, "text", "")
                        offsets = getattr(decoded, "word_offsets", [])
                    except Exception:
                        txt = self.processor.batch_decode(logits.detach().cpu().numpy()).text[0]
                        offsets = []
                else:
                    pred_cpu = pred_ids.cpu() if hasattr(pred_ids, "cpu") else pred_ids
                    txt = self.processor.decode(pred_cpu)
                    try:
                        if use_timestamps == "char":
                            decoded = self.processor.decode(pred_cpu, output_char_offsets=True)
                            offsets = getattr(decoded, "char_offsets", [])
                        else:
                            decoded = self.processor.decode(pred_cpu, output_word_offsets=True)
                            offsets = getattr(decoded, "word_offsets", [])
                            if not offsets:
                                decoded_char = self.processor.decode(pred_cpu, output_char_offsets=True)
                                offsets = getattr(decoded_char, "char_offsets", [])
                    except Exception:
                        offsets = []

                for item in offsets:
                    c_text = item.get("word") or item.get("char") or ""
                    c_text = c_text.strip()
                    if not c_text:
                        continue
                    s_time = round(item.get("start_offset", 0) * time_offset, 2)
                    e_time = round(item.get("end_offset", 0) * time_offset, 2)
                    chunks.append({
                        "text": c_text,
                        "timestamp": (s_time, e_time),
                        "start": s_time,
                        "end": e_time,
                    })

                return {
                    "text": txt,
                    "chunks": chunks,
                    "timestamps": chunks,
                }
            else:
                if self.model_name == "airesearch/wav2vec2-large-xlsr-53-th":
                    txt = self.processor.decode(pred_ids)
                elif self.lm:
                    txt = self.processor.batch_decode(logits.detach().numpy()).text[0]
                else:
                    txt = self.processor.decode(pred_ids)
                return txt

_model_name = "typhoon_asr"
_model = None


def asr(
    data: Union[str, np.ndarray],
    model: str = _model_name,
    lm: bool = False,
    device: str = None,
    sampling_rate: int = 16_000,
    return_timestamps: Optional[Union[bool, str]] = None,
    timestamps: Optional[Union[bool, str]] = None,
) -> Union[str, dict]:
    """
    :param Union[str, np.ndarray] data: path of sound file or numpy array of the voice
    :param str model: The ASR model name (default: "typhoon_asr")
    :param bool lm: Use language model (for wav2vec2 models with LM)
    :param str device: device ("cpu", "cuda", "auto")
    :param int sampling_rate: The sample rate
    :param Optional[Union[bool, str]] return_timestamps: If True, returns a dictionary with
        text, chunks, and timestamps. Can also be "word" or "char".
    :param Optional[Union[bool, str]] timestamps: Alias for return_timestamps.
    :return: Thai text string from ASR or dictionary with timestamps.
    :rtype: Union[str, dict]

    **Options for model**
        * *typhoon_asr* / *typhoon-asr-realtime* (default) - Typhoon FastConformer RNN-T ONNX model
        * *airesearch/wav2vec2-large-xlsr-53-th* - AI RESEARCH - PyThaiNLP model (requires pythaiasr[torch])
        * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm* - Thai Wav2Vec2 with CommonVoice V8 (newmm tokenizer) (+ language model, requires pythaiasr[torch])
        * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut* - Thai Wav2Vec2 with CommonVoice V8 (deepcut tokenizer) (+ language model, requires pythaiasr[torch])
        * *biodatlab/whisper-small-th-combined* - Thai Whisper small model (requires pythaiasr[torch])
        * *biodatlab/whisper-th-medium-combined* - Thai Whisper medium model (requires pythaiasr[torch])
        * *biodatlab/whisper-th-large-combined* - Thai Whisper large model (requires pythaiasr[torch])
        * *biodatlab/whisper-th-medium-timestamp* - Thai Whisper medium model with timestamp support (requires pythaiasr[torch])
    """
    global _model, _model_name
    if model != _model_name or _model is None:
        _model = ASR(model, lm=lm, device=device)
        _model_name = model

    return _model(
        data=data,
        sampling_rate=sampling_rate,
        return_timestamps=return_timestamps,
        timestamps=timestamps,
    )


def stream_asr(
    model: str = _model_name,
    lm: bool = False,
    device: str = None, 
    chunk_duration: float = None,
    sampling_rate: int = 16_000,
    return_timestamps: bool = False,
    timestamps: Optional[bool] = None,
):
    """
    Stream audio from microphone/soundcard and perform real-time ASR.
    
    :param str model: The ASR model name (default: "typhoon_asr")
    :param bool lm: Use language model (for wav2vec2 models with LM)
    :param str device: device
    :param float chunk_duration: Duration of each audio chunk in seconds (default: 0.48s for Typhoon, 5.0s for others)
    :param int sampling_rate: The sample rate (default: 16000)
    :param bool return_timestamps: If True, yields dict with text and timestamp tuple for each chunk
    :param Optional[bool] timestamps: Alias for return_timestamps
    :yield: Thai text transcription (or dict with timestamp) from each audio chunk
    
    **Options for model**
        * *typhoon_asr* / *typhoon-asr-realtime* (default) - Typhoon FastConformer RNN-T ONNX model (recommended for streaming)
        * *airesearch/wav2vec2-large-xlsr-53-th* - AI RESEARCH - PyThaiNLP model (requires pythaiasr[torch])
        * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm* - Thai Wav2Vec2 with CommonVoice V8 (newmm tokenizer) (+ language model, requires pythaiasr[torch])
        * *wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut* - Thai Wav2Vec2 with CommonVoice V8 (deepcut tokenizer) (+ language model, requires pythaiasr[torch])
        * *biodatlab/whisper-small-th-combined* - Thai Whisper small model (requires pythaiasr[torch])
        * *biodatlab/whisper-th-medium-combined* - Thai Whisper medium model (requires pythaiasr[torch])
        * *biodatlab/whisper-th-large-combined* - Thai Whisper large model (requires pythaiasr[torch])
        * *biodatlab/whisper-th-medium-timestamp* - Thai Whisper medium model with timestamp support (requires pythaiasr[torch])
    
    **Example:**
        .. code-block:: python
        
            from pythaiasr import stream_asr
            
            # Stream audio with Typhoon ASR and print transcriptions
            for transcription in stream_asr(model="typhoon_asr"):
                print(transcription)
                # Press Ctrl+C to stop
    """
    try:
        import pyaudio
    except ImportError:
        raise ImportError(
            "pyaudio is required for audio streaming. "
            "Install it with: pip install pyaudio"
        )
    
    global _model, _model_name
    if model != _model_name or _model is None:
        _model = ASR(model, lm=lm, device=device)
        _model_name = model
    
    if chunk_duration is None:
        chunk_duration = 0.48 if _model.is_typhoon else 5.0

    use_timestamps = return_timestamps or bool(timestamps)

    # If Typhoon model, use stateful RealtimeStreamASR
    streamer = None
    if _model.is_typhoon:
        streamer = RealtimeStreamASR(
            model=_model.model,
            sample_rate=sampling_rate,
            step_sec=chunk_duration,
        )

    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    # Calculate chunk size
    chunk_size = int(sampling_rate * chunk_duration)
    stream_elapsed = 0.0
    
    try:
        # Open stream
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sampling_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        print(f"Recording audio from microphone... (chunk duration: {chunk_duration}s)")
        print("Press Ctrl+C to stop.")
        
        while True:
            audio_data = stream.read(chunk_size, exception_on_overflow=False)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            chunk_time_start = round(stream_elapsed, 2)
            stream_elapsed += len(audio_array) / float(sampling_rate)
            chunk_time_end = round(stream_elapsed, 2)
            
            if streamer is not None:
                transcription = streamer.process_chunk(audio_array)
            else:
                transcription = _model(data=audio_array, sampling_rate=sampling_rate)
            
            if transcription and transcription.strip():
                if use_timestamps:
                    yield {
                        "text": transcription,
                        "timestamp": (chunk_time_start, chunk_time_end),
                        "start": chunk_time_start,
                        "end": chunk_time_end,
                    }
                else:
                    yield transcription
                
    except KeyboardInterrupt:
        print("\nStopping audio stream...")
    finally:
        if streamer is not None:
            trailing = streamer.flush()
            if trailing and trailing.strip():
                if use_timestamps:
                    yield {
                        "text": trailing,
                        "timestamp": (round(stream_elapsed, 2), round(stream_elapsed, 2)),
                        "start": round(stream_elapsed, 2),
                        "end": round(stream_elapsed, 2),
                    }
                else:
                    yield trailing

        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        audio.terminate()


__all__ = [
    "ASR",
    "asr",
    "stream_asr",
    "Diarization",
    "diarize",
    "asr_diarize",
    "merge_same_speaker_segments",
    "FastConformerRNNT",
    "TyphoonASR",
    "RealtimeStreamASR",
    "StreamingTranscriber",
    "stream_from_mic",
    "stream_from_file",
    "list_audio_devices",
    "extract_features",
    "load_audio",
    "get_pythaiasr_path",
    "get_typhoon_model_files",
    "get_diarization_model_files",
    "download_file",
]
