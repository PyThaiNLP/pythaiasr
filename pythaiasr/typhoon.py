# -*- coding: utf-8 -*-
"""Typhoon FastConformer RNN-T ONNX Speech Recognition for PyThaiASR.

Supports:
- Offline audio file & array transcription
- Real-time sliding context streaming transcription (chunk-by-chunk)
- Automatic download of ONNX models to ~/pythaiasr-data/typhoon-asr-realtime/
"""

from __future__ import annotations

import json
import math
import os
import queue
import sys
import time
import wave
from pathlib import Path
from typing import List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None

from pythaiasr.download import get_typhoon_model_files


# ==============================================================================
# 1. Mel Spectrogram Preprocessor (NeMo AudioToMelSpectrogramPreprocessor equivalent)
# ==============================================================================

_MEL_FILTERBANK_CACHE = {}


def hz_to_mel_slaney(frequencies: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Hz to Mel scale using Slaney's formula."""
    f_min = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp  # 15.0
    logstep = np.log(6.4) / 27.0

    is_scalar = np.isscalar(frequencies) or np.ndim(frequencies) == 0
    freq_arr = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    mels = (freq_arr - f_min) / f_sp

    mask = freq_arr >= min_log_hz
    if np.any(mask):
        mels[mask] = min_log_mel + np.log(freq_arr[mask] / min_log_hz) / logstep

    return float(mels[0]) if is_scalar else mels


def mel_to_hz_slaney(mels: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Mel scale to Hz using Slaney's formula."""
    f_min = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp  # 15.0
    logstep = np.log(6.4) / 27.0

    is_scalar = np.isscalar(mels) or np.ndim(mels) == 0
    mel_arr = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    freqs = f_min + f_sp * mel_arr

    mask = mel_arr >= min_log_mel
    if np.any(mask):
        freqs[mask] = min_log_hz * np.exp(logstep * (mel_arr[mask] - min_log_mel))

    return float(freqs[0]) if is_scalar else freqs


def create_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Create Slaney-style Mel filterbank matrix of shape (n_mels, 1 + n_fft // 2)."""
    if fmax is None:
        fmax = sr / 2.0

    cache_key = (sr, n_fft, n_mels, float(fmin), float(fmax))
    if cache_key in _MEL_FILTERBANK_CACHE:
        return _MEL_FILTERBANK_CACHE[cache_key]

    try:
        import librosa
        fb = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, norm="slaney"
        ).astype(np.float32)
        _MEL_FILTERBANK_CACHE[cache_key] = fb
        return fb
    except ImportError:
        pass

    fft_freqs = np.linspace(0.0, sr / 2.0, int(1 + n_fft // 2), endpoint=True)
    mel_min = hz_to_mel_slaney(fmin)
    mel_max = hz_to_mel_slaney(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz_slaney(mel_points)

    fdiff = np.diff(hz_points)
    ramps = np.subtract.outer(hz_points, fft_freqs)

    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney normalization: normalize each filter's area to 2 / (f[i+2] - f[i])
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    weights *= enorm[:, np.newaxis]
    fb = weights.astype(np.float32)
    _MEL_FILTERBANK_CACHE[cache_key] = fb
    return fb


def compute_stft_numpy(
    signal: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """Compute STFT power spectrum using NumPy matching PyTorch / NeMo."""
    signal = np.asarray(signal, dtype=np.float32).flatten()

    # Periodic=False Hann window of win_length
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(win_length) / (win_length - 1)))

    # Window centered in n_fft buffer
    padded_window = np.zeros(n_fft, dtype=np.float64)
    start = (n_fft - win_length) // 2
    padded_window[start : start + win_length] = window

    # Center padding using reflect mode
    pad_amount = n_fft // 2
    if len(signal) < pad_amount:
        signal = np.pad(signal, (0, pad_amount - len(signal)), mode="constant")
    padded_signal = np.pad(signal, (pad_amount, pad_amount), mode="reflect")

    # Frame extraction
    try:
        all_frames = np.lib.stride_tricks.sliding_window_view(padded_signal, window_shape=n_fft)
        frames = all_frames[::hop_length]
    except AttributeError:
        num_frames = 1 + (len(padded_signal) - n_fft) // hop_length
        frames = np.lib.stride_tricks.as_strided(
            padded_signal,
            shape=(num_frames, n_fft),
            strides=(padded_signal.strides[0] * hop_length, padded_signal.strides[0]),
        )

    windowed_frames = frames * padded_window
    spec = np.fft.rfft(windowed_frames, n=n_fft, axis=1)
    power_spec = (np.abs(spec) ** 2).T
    return power_spec.astype(np.float32)


def extract_features(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 512,
    win_length: int = 400,
    hop_length: int = 160,
    preemph: float = 0.97,
    pad_to: int = 16,
) -> Tuple[np.ndarray, int]:
    """Extract 80-channel log-mel spectrogram features matching NeMo FastConformer."""
    # 1. Pre-emphasis filter
    if preemph > 0:
        audio = np.append(audio[0], audio[1:] - preemph * audio[:-1])

    # 2. Compute STFT power spectrum
    power_spec = compute_stft_numpy(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # 3. Apply Mel filterbank
    fb = create_mel_filterbank(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sample_rate / 2.0)
    mel_spec = np.matmul(fb, power_spec)

    # 4. Log compression with zero guard
    mel_spec = np.log(mel_spec + (2.0 ** -24))

    # 5. Per-feature normalization
    seq_len = mel_spec.shape[1]
    mean = np.mean(mel_spec, axis=1, keepdims=True)
    std = np.std(mel_spec, axis=1, keepdims=True)
    norm_mel = (mel_spec - mean) / (std + 1e-5)

    # 6. Pad time dimension to multiple of pad_to
    if pad_to > 0 and (norm_mel.shape[1] % pad_to != 0):
        pad_amount = pad_to - (norm_mel.shape[1] % pad_to)
        norm_mel = np.pad(norm_mel, ((0, 0), (0, pad_amount)), mode="constant", constant_values=0.0)

    features = np.expand_dims(norm_mel, axis=0).astype(np.float32)
    return features, seq_len


# ==============================================================================
# 2. Audio File Loader
# ==============================================================================

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D float32 audio from orig_sr to target_sr using librosa, scipy, or linear interpolation."""
    if orig_sr == target_sr:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        pass

    try:
        from scipy import signal
        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        return signal.resample_poly(audio, up, down).astype(np.float32)
    except (ImportError, AttributeError):
        pass

    # Fallback to linear interpolation
    num_samples = int(len(audio) * target_sr / orig_sr)
    indices = np.linspace(0, len(audio) - 1, num_samples)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def load_audio(path: Union[str, Path], target_sr: int = 16000) -> np.ndarray:
    """Load audio file as single-channel (mono) float32 array at target sample rate."""
    path = str(path)

    # Option A: soundfile
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
        return audio
    except ImportError:
        pass

    # Option B: librosa
    try:
        import librosa
        audio, _ = librosa.load(path, sr=target_sr, mono=True)
        return audio.astype(np.float32)
    except ImportError:
        pass

    # Option C: scipy.io.wavfile
    try:
        from scipy.io import wavfile
        sr, audio = wavfile.read(path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128.0) / 128.0
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
        return audio.astype(np.float32)
    except ImportError:
        pass

    # Option D: wave module
    if path.lower().endswith(".wav"):
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

            if sampwidth == 2:
                data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise RuntimeError(f"Unsupported sample width: {sampwidth} bytes")

            if n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=1)
            if framerate != target_sr:
                data = resample_audio(data, framerate, target_sr)
            return data

    raise RuntimeError(
        "Could not load audio. Please install soundfile (`pip install soundfile`) or librosa (`pip install librosa`)."
    )


# ==============================================================================
# 3. FastConformer RNN-T ONNX Model Wrapper
# ==============================================================================

class FastConformerRNNT:
    """End-to-end inference pipeline for Typhoon FastConformer RNN-T ONNX models."""

    def __init__(
        self,
        encoder_path: Optional[Union[str, Path]] = None,
        decoder_path: Optional[Union[str, Path]] = None,
        vocab_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        max_symbols_per_step: int = 10,
        model_dir: Optional[str] = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for Typhoon ASR. Please install it using:\n"
                "  pip install onnxruntime        # For CPU / Apple Silicon\n"
                "  pip install onnxruntime-gpu    # For NVIDIA CUDA\n"
                "Or install pythaiasr with: pip install pythaiasr[typhoon]"
            )

        # If any path is not provided, download or load from ~/pythaiasr-data/typhoon-asr-realtime/
        if encoder_path is None or decoder_path is None or vocab_path is None:
            def_enc, def_dec, def_voc = get_typhoon_model_files(model_dir=model_dir)
            encoder_path = encoder_path or def_enc
            decoder_path = decoder_path or def_dec
            vocab_path = vocab_path or def_voc

        self.encoder_path = Path(encoder_path)
        self.decoder_path = Path(decoder_path)
        self.vocab_path = Path(vocab_path)
        self.max_symbols_per_step = max_symbols_per_step

        if not self.encoder_path.exists():
            raise FileNotFoundError(f"Encoder model not found: {self.encoder_path}")
        if not self.decoder_path.exists():
            raise FileNotFoundError(f"Decoder model not found: {self.decoder_path}")
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")

        # Load vocabulary
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.vocab: List[str] = json.load(f)

        self.blank_id = len(self.vocab)
        self.hidden_dim = 640

        providers = self._resolve_providers(device, ort)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.encoder_session = ort.InferenceSession(str(self.encoder_path), opts, providers=providers)
        self.decoder_session = ort.InferenceSession(str(self.decoder_path), opts, providers=providers)

    @staticmethod
    def _resolve_providers(device: str, ort) -> List[str]:
        available = ort.get_available_providers()
        if device == "cuda":
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            raise RuntimeError("CUDA requested, but 'CUDAExecutionProvider' is not available in onnxruntime.")
        elif device == "coreml":
            if "CoreMLExecutionProvider" in available:
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            raise RuntimeError("CoreML requested, but 'CoreMLExecutionProvider' is not available in onnxruntime.")
        elif device == "cpu":
            return ["CPUExecutionProvider"]
        else:  # auto
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CoreMLExecutionProvider" in available:
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]

    def decode_token_ids(self, token_ids: List[int]) -> str:
        """Decode a sequence of vocabulary token IDs into readable Thai text."""
        pieces = []
        for tid in token_ids:
            if 0 <= tid < len(self.vocab):
                token_str = self.vocab[tid]
                if token_str in {"<unk>", "<s>", "</s>", "<pad>", "<bos>", "<eos>"}:
                    continue
                pieces.append(token_str)
        text = "".join(pieces)
        text = text.replace("\u2581", " ").strip()
        return text

    def greedy_decode(
        self,
        encoder_outputs: np.ndarray,
        encoded_lengths: np.ndarray,
    ) -> List[dict]:
        """Perform autoregressive RNN-T greedy decoding over encoder outputs."""
        batch_size = encoder_outputs.shape[0]
        results = []

        for b in range(batch_size):
            t_len = int(encoded_lengths[b])
            enc_b = encoder_outputs[b : b + 1]

            h = np.zeros((1, 1, self.hidden_dim), dtype=np.float32)
            c = np.zeros((1, 1, self.hidden_dim), dtype=np.float32)

            last_token = self.blank_id
            emitted_tokens: List[int] = []
            token_timestamps: List[int] = []

            for t in range(t_len):
                enc_frame = enc_b[:, :, t : t + 1]
                symbols_added = 0
                while symbols_added < self.max_symbols_per_step:
                    targets = np.array([[last_token]], dtype=np.int32)
                    target_length = np.array([1], dtype=np.int32)

                    outputs, _, next_h, next_c = self.decoder_session.run(
                        None,
                        {
                            "encoder_outputs": enc_frame,
                            "targets": targets,
                            "target_length": target_length,
                            "input_states_1": h,
                            "input_states_2": c,
                        },
                    )

                    logits = outputs[0, 0, 0, :]
                    predicted_token = int(np.argmax(logits))

                    if predicted_token == self.blank_id:
                        break
                    else:
                        emitted_tokens.append(predicted_token)
                        token_timestamps.append(t)
                        last_token = predicted_token
                        h = next_h
                        c = next_c
                        symbols_added += 1

            text = self.decode_token_ids(emitted_tokens)
            results.append(
                {
                    "tokens": emitted_tokens,
                    "timestamps": token_timestamps,
                    "text": text,
                }
            )

        return results

    def tokens_to_timestamp_chunks(
        self,
        emitted_tokens: List[int],
        token_timestamps: List[int],
        sample_rate: int = 16000,
        mode: Union[bool, str] = True,
    ) -> List[dict]:
        """Convert emitted tokens and their frame timestamps into timestamped chunks.

        :param emitted_tokens: List of emitted vocabulary token IDs.
        :param token_timestamps: List of encoder frame indices for each token.
        :param sample_rate: Audio sampling rate in Hz (default: 16000).
        :param mode: True, "word" for word/segment chunks, or "char"/"token" for character-level chunks.
        :return: List of chunk dictionaries containing text, timestamp tuple, start, and end in seconds.
        """
        if not emitted_tokens or not token_timestamps:
            return []

        # 1280 audio samples per encoder frame (hop_length=160, 8x FastConformer subsampling)
        frame_duration = 1280.0 / float(sample_rate)
        special_tokens = {"<unk>", "<s>", "</s>", "<pad>", "<bos>", "<eos>"}

        items = []
        for tid, t in zip(emitted_tokens, token_timestamps):
            if 0 <= tid < len(self.vocab):
                tok_str = self.vocab[tid]
                if tok_str in special_tokens:
                    continue
                items.append((tok_str, t))

        if not items:
            return []

        # Character / token level
        if mode in ("char", "token"):
            char_chunks = []
            for tok_str, t in items:
                piece = tok_str.replace("\u2581", " ").strip()
                if not piece:
                    continue
                s_time = round(t * frame_duration, 2)
                e_time = round((t + 1) * frame_duration, 2)
                char_chunks.append({
                    "text": piece,
                    "timestamp": (s_time, e_time),
                    "start": s_time,
                    "end": e_time,
                })
            return char_chunks

        full_text = self.decode_token_ids(emitted_tokens)

        # Word level with PyThaiNLP if requested and available
        if mode == "word":
            try:
                from pythainlp.tokenize import word_tokenize
                words = [w for w in word_tokenize(full_text, engine="newmm") if w.strip()]
                if len(words) > 1:
                    char_times = []
                    for tok_str, t in items:
                        cleaned = tok_str.replace("\u2581", " ")
                        for ch in cleaned:
                            char_times.append((ch, t))

                    word_chunks = []
                    idx = 0
                    for word in words:
                        while idx < len(char_times) and char_times[idx][0].isspace():
                            idx += 1
                        if idx >= len(char_times):
                            break
                        word_chars = [ch for ch in word if not ch.isspace()]
                        matched_ts = []
                        for w_ch in word_chars:
                            while idx < len(char_times) and char_times[idx][0] != w_ch:
                                idx += 1
                            if idx < len(char_times) and char_times[idx][0] == w_ch:
                                matched_ts.append(char_times[idx][1])
                                idx += 1
                        if matched_ts:
                            w_start = round(min(matched_ts) * frame_duration, 2)
                            w_end = round((max(matched_ts) + 1) * frame_duration, 2)
                            word_chunks.append({
                                "text": word,
                                "timestamp": (w_start, w_end),
                                "start": w_start,
                                "end": w_end,
                            })
                    if word_chunks:
                        return word_chunks
            except ImportError:
                pass

        # Standard segment/word chunking (based on SentencePiece space tokens and pause detection)
        chunks = []
        curr_tokens = []
        curr_start_t = None
        curr_end_t = None

        for tok_str, t in items:
            is_new = False
            if curr_start_t is None:
                is_new = True
            elif tok_str.startswith("\u2581") or tok_str.startswith(" "):
                is_new = True
            elif curr_end_t is not None and (t - curr_end_t) >= 6:  # Pause >= 0.48s
                is_new = True

            if is_new and curr_tokens:
                chunk_text = "".join(curr_tokens).replace("\u2581", " ").strip()
                if chunk_text:
                    s_time = round(curr_start_t * frame_duration, 2)
                    e_time = round((curr_end_t + 1) * frame_duration, 2)
                    chunks.append({
                        "text": chunk_text,
                        "timestamp": (s_time, e_time),
                        "start": s_time,
                        "end": e_time,
                    })
                curr_tokens = []
                curr_start_t = t

            if curr_start_t is None:
                curr_start_t = t
            curr_tokens.append(tok_str)
            curr_end_t = t

        if curr_tokens:
            chunk_text = "".join(curr_tokens).replace("\u2581", " ").strip()
            if chunk_text:
                s_time = round(curr_start_t * frame_duration, 2)
                e_time = round((curr_end_t + 1) * frame_duration, 2)
                chunks.append({
                    "text": chunk_text,
                    "timestamp": (s_time, e_time),
                    "start": s_time,
                    "end": e_time,
                })

        return chunks

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        sample_rate: int = 16000,
        return_timestamps: Optional[Union[bool, str]] = None,
        timestamps: Optional[Union[bool, str]] = None,
    ) -> Union[str, dict]:
        """Transcribe an audio file or audio array to Thai text.

        :param audio_input: Path to audio file or float32 numpy audio array.
        :param int sample_rate: Sample rate of the audio (default: 16000).
        :param Optional[Union[bool, str]] return_timestamps: If True, returns a dictionary with
            text, chunks, and timestamps. Also accepts "word" or "char".
        :param Optional[Union[bool, str]] timestamps: Alias for return_timestamps.
        :return: Thai text string (default) or dictionary with timestamps.
        """
        use_timestamps = return_timestamps if return_timestamps is not None else timestamps
        if isinstance(audio_input, (str, Path)):
            audio = load_audio(audio_input, target_sr=sample_rate)
        else:
            audio = np.asarray(audio_input, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

        features, _ = extract_features(audio, sample_rate=sample_rate)
        length = np.array([features.shape[-1]], dtype=np.int64)

        encoder_outputs, encoded_lengths = self.encoder_session.run(
            None,
            {"audio_signal": features, "length": length},
        )

        results = self.greedy_decode(encoder_outputs, encoded_lengths)
        result = results[0]
        text = result["text"]

        if use_timestamps:
            chunks = self.tokens_to_timestamp_chunks(
                emitted_tokens=result.get("tokens", []),
                token_timestamps=result.get("timestamps", []),
                sample_rate=sample_rate,
                mode=use_timestamps,
            )
            return {
                "text": text,
                "chunks": chunks,
                "timestamps": chunks,
            }

        return text

    def create_streaming_session(
        self,
        step_sec: float = 0.48,
        left_context_sec: float = 0.64,
        right_context_sec: float = 0.32,
    ) -> StreamingTranscriber:
        """Create a stateful streaming session for real-time chunk transcription."""
        return StreamingTranscriber(
            self,
            step_sec=step_sec,
            left_context_sec=left_context_sec,
            right_context_sec=right_context_sec,
        )


# ==============================================================================
# 4. Stateful Streaming Transcribers
# ==============================================================================

class StreamingTranscriber:
    """Low-latency streaming transcriber with sliding acoustic context window."""

    def __init__(
        self,
        model: FastConformerRNNT,
        sample_rate: int = 16000,
        step_sec: float = 0.48,
        left_context_sec: float = 0.64,
        right_context_sec: float = 0.32,
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.step_sec = step_sec
        self.left_context_sec = left_context_sec
        self.right_context_sec = right_context_sec

        self.step_samples = int(sample_rate * step_sec)
        self.left_samples = int(sample_rate * left_context_sec)
        self.right_samples = int(sample_rate * right_context_sec)

        self.h = np.zeros((1, 1, model.hidden_dim), dtype=np.float32)
        self.c = np.zeros((1, 1, model.hidden_dim), dtype=np.float32)
        self.last_token = model.blank_id

        self.audio_history = np.zeros(0, dtype=np.float32)
        self.cursor_sample = 0
        self.all_emitted_tokens: List[int] = []

    def reset(self):
        """Reset streaming state."""
        self.h.fill(0)
        self.c.fill(0)
        self.last_token = self.model.blank_id
        self.audio_history = np.zeros(0, dtype=np.float32)
        self.cursor_sample = 0
        self.all_emitted_tokens.clear()

    def feed_chunk(self, chunk: np.ndarray) -> str:
        """Feed audio chunk and return newly emitted text segment."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim > 1:
            chunk = np.mean(chunk, axis=1)

        self.audio_history = np.concatenate([self.audio_history, chunk])

        new_text_segments = []
        while (len(self.audio_history) - self.cursor_sample) >= (self.step_samples + self.right_samples):
            text = self._decode_window(is_flush=False)
            if text:
                new_text_segments.append(text)

        max_history = self.left_samples + self.sample_rate * 3
        if self.cursor_sample > max_history:
            trim = self.cursor_sample - self.left_samples
            self.audio_history = self.audio_history[trim:]
            self.cursor_sample -= trim

        return " ".join(new_text_segments)

    def flush(self) -> str:
        """Process any remaining audio in the buffer."""
        new_text_segments = []
        while self.cursor_sample < len(self.audio_history):
            text = self._decode_window(is_flush=True)
            if text:
                new_text_segments.append(text)
        return " ".join(new_text_segments)

    def _decode_window(self, is_flush: bool = False) -> str:
        cursor = self.cursor_sample
        total_len = len(self.audio_history)
        if cursor >= total_len:
            return ""

        win_start = max(0, cursor - self.left_samples)
        if is_flush:
            win_end = total_len
            step_end = total_len
        else:
            win_end = min(total_len, cursor + self.step_samples + self.right_samples)
            step_end = min(total_len, cursor + self.step_samples)

        window = self.audio_history[win_start:win_end]
        curr_start_sample = cursor - win_start
        curr_end_sample = step_end - win_start
        self.cursor_sample = step_end

        features, _ = extract_features(window, sample_rate=self.sample_rate)
        length = np.array([features.shape[-1]], dtype=np.int64)

        encoder_outputs, encoded_lengths = self.model.encoder_session.run(
            None,
            {"audio_signal": features, "length": length},
        )

        total_enc_frames = int(encoded_lengths[0])
        start_frame = int(round(curr_start_sample / 1280.0))
        end_frame = int(round(curr_end_sample / 1280.0))
        end_frame = min(end_frame, total_enc_frames)

        new_tokens = []
        for t in range(start_frame, end_frame):
            enc_frame = encoder_outputs[:, :, t : t + 1]
            symbols_added = 0
            while symbols_added < self.model.max_symbols_per_step:
                targets = np.array([[self.last_token]], dtype=np.int32)
                target_length = np.array([1], dtype=np.int32)

                outputs, _, next_h, next_c = self.model.decoder_session.run(
                    None,
                    {
                        "encoder_outputs": enc_frame,
                        "targets": targets,
                        "target_length": target_length,
                        "input_states_1": self.h,
                        "input_states_2": self.c,
                    },
                )

                logits = outputs[0, 0, 0, :]
                pred_token = int(np.argmax(logits))

                if pred_token == self.model.blank_id:
                    break
                else:
                    new_tokens.append(pred_token)
                    self.all_emitted_tokens.append(pred_token)
                    self.last_token = pred_token
                    self.h = next_h
                    self.c = next_c
                    symbols_added += 1

        if not new_tokens:
            return ""

        return self.model.decode_token_ids(new_tokens)

    def get_full_text(self) -> str:
        """Return the accumulated transcription of the entire stream so far."""
        return self.model.decode_token_ids(self.all_emitted_tokens)


class RealtimeStreamASR:
    """Stream processor for low-latency FastConformer RNN-T inference with energy-based silence gating."""

    def __init__(
        self,
        model: FastConformerRNNT,
        sample_rate: int = 16000,
        step_sec: float = 0.48,
        left_context_sec: float = 0.64,
        right_context_sec: float = 0.32,
        silence_threshold: float = 0.005,
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.step_sec = step_sec
        self.left_context_sec = left_context_sec
        self.right_context_sec = right_context_sec
        self.silence_threshold = silence_threshold

        self.step_samples = int(sample_rate * step_sec)
        self.left_samples = int(sample_rate * left_context_sec)
        self.right_samples = int(sample_rate * right_context_sec)

        self.h = np.zeros((1, 1, model.hidden_dim), dtype=np.float32)
        self.c = np.zeros((1, 1, model.hidden_dim), dtype=np.float32)
        self.last_token = model.blank_id

        self.audio_history = np.zeros(0, dtype=np.float32)
        self.cursor_sample = 0
        self.all_emitted_tokens: List[int] = []

    def reset(self):
        """Reset internal decoder state and buffers."""
        self.h.fill(0)
        self.c.fill(0)
        self.last_token = self.model.blank_id
        self.audio_history = np.zeros(0, dtype=np.float32)
        self.cursor_sample = 0
        self.all_emitted_tokens.clear()

    def process_chunk(self, chunk: np.ndarray) -> str:
        """Feed incoming audio chunk and return newly emitted text segment."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim > 1:
            chunk = np.mean(chunk, axis=1)

        self.audio_history = np.concatenate([self.audio_history, chunk])

        new_text_segments = []
        while (len(self.audio_history) - self.cursor_sample) >= (self.step_samples + self.right_samples):
            text = self._decode_window(is_flush=False)
            if text:
                new_text_segments.append(text)

        max_history = self.left_samples + self.sample_rate * 3
        if self.cursor_sample > max_history:
            trim = self.cursor_sample - self.left_samples
            self.audio_history = self.audio_history[trim:]
            self.cursor_sample -= trim

        return " ".join(new_text_segments)

    def flush(self) -> str:
        """Process any remaining audio at the end of the stream."""
        new_text_segments = []
        while self.cursor_sample < len(self.audio_history):
            text = self._decode_window(is_flush=True)
            if text:
                new_text_segments.append(text)
        return " ".join(new_text_segments)

    def _decode_window(self, is_flush: bool = False) -> str:
        cursor = self.cursor_sample
        total_len = len(self.audio_history)

        if cursor >= total_len:
            return ""

        win_start = max(0, cursor - self.left_samples)
        if is_flush:
            win_end = total_len
            step_end = total_len
        else:
            win_end = min(total_len, cursor + self.step_samples + self.right_samples)
            step_end = min(total_len, cursor + self.step_samples)

        window = self.audio_history[win_start:win_end]

        step_audio = self.audio_history[cursor:step_end]
        rms = float(np.sqrt(np.mean(step_audio ** 2))) if len(step_audio) > 0 else 0.0

        curr_start_sample = cursor - win_start
        curr_end_sample = step_end - win_start
        self.cursor_sample = step_end

        if self.silence_threshold > 0 and rms < self.silence_threshold:
            return ""

        features, _ = extract_features(window, sample_rate=self.sample_rate)
        length = np.array([features.shape[-1]], dtype=np.int64)

        encoder_outputs, encoded_lengths = self.model.encoder_session.run(
            None,
            {"audio_signal": features, "length": length},
        )

        total_enc_frames = int(encoded_lengths[0])
        start_frame = int(round(curr_start_sample / 1280.0))
        end_frame = int(round(curr_end_sample / 1280.0))
        end_frame = min(end_frame, total_enc_frames)

        new_tokens = []
        for t in range(start_frame, end_frame):
            enc_frame = encoder_outputs[:, :, t : t + 1]
            symbols_added = 0
            while symbols_added < self.model.max_symbols_per_step:
                targets = np.array([[self.last_token]], dtype=np.int32)
                target_length = np.array([1], dtype=np.int32)

                outputs, _, next_h, next_c = self.model.decoder_session.run(
                    None,
                    {
                        "encoder_outputs": enc_frame,
                        "targets": targets,
                        "target_length": target_length,
                        "input_states_1": self.h,
                        "input_states_2": self.c,
                    },
                )

                logits = outputs[0, 0, 0, :]
                pred_token = int(np.argmax(logits))

                if pred_token == self.model.blank_id:
                    break
                else:
                    new_tokens.append(pred_token)
                    self.all_emitted_tokens.append(pred_token)
                    self.last_token = pred_token
                    self.h = next_h
                    self.c = next_c
                    symbols_added += 1

        if not new_tokens:
            return ""

        return self.model.decode_token_ids(new_tokens)

    def get_full_transcript(self) -> str:
        """Return accumulated transcript of all emitted tokens."""
        return self.model.decode_token_ids(self.all_emitted_tokens)


# ==============================================================================
# 5. Microphone and File Streaming Utilities
# ==============================================================================

def stream_from_mic(
    streamer: RealtimeStreamASR,
    device_index: Optional[int] = None,
    chunk_sec: float = 0.48,
):
    """Capture live microphone audio and print real-time transcription to console."""
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError(
            "sounddevice is required for microphone streaming. "
            "Please install it with: pip install sounddevice"
        )

    sample_rate = streamer.sample_rate
    block_size = int(sample_rate * chunk_sec)
    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"\n[Audio Warning: {status}]", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    print("\n" + "=" * 60)
    print(" LIVE REAL-TIME ASR (Press Ctrl+C to Stop)")
    print(f" Sample rate: {sample_rate} Hz | Chunk size: {chunk_sec}s")
    if device_index is not None:
        print(f" Input Device Index: {device_index}")
    print(" Speak into your microphone...")
    print("=" * 60 + "\n")

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            device=device_index,
            callback=mic_callback,
        ):
            while True:
                try:
                    chunk = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                text = streamer.process_chunk(chunk)
                if text:
                    print(text, end=" ", flush=True)

    except KeyboardInterrupt:
        pass

    trailing_text = streamer.flush()
    if trailing_text:
        print(trailing_text, end=" ", flush=True)

    print("\n\n--- Session Finished ---")
    full_text = streamer.get_full_transcript()
    print("Final Transcript:")
    print(full_text if full_text else "(No speech detected)")
    print("-------------------------\n")


def stream_from_file(
    streamer: RealtimeStreamASR,
    file_path: Union[str, Path],
    chunk_sec: float = 0.48,
    simulate_realtime: bool = True,
):
    """Stream an existing audio file chunk-by-chunk to simulate live real-time input."""
    audio = load_audio(file_path, target_sr=streamer.sample_rate)
    chunk_samples = int(streamer.sample_rate * chunk_sec)
    total_samples = len(audio)
    duration_sec = total_samples / streamer.sample_rate

    print("\n" + "=" * 60)
    print(f" STREAMING FROM FILE: {file_path}")
    print(f" Duration: {duration_sec:.2f}s | Chunk: {chunk_sec}s | Real-time pacing: {simulate_realtime}")
    print("=" * 60 + "\n")

    start_time = time.time()
    num_chunks = int(math.ceil(total_samples / chunk_samples))

    try:
        for i in range(num_chunks):
            t_chunk_start = time.time()
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples, total_samples)
            chunk = audio[start_idx:end_idx]

            text = streamer.process_chunk(chunk)
            if text:
                print(text, end=" ", flush=True)

            if simulate_realtime:
                elapsed = time.time() - t_chunk_start
                sleep_time = chunk_sec - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")

    trailing_text = streamer.flush()
    if trailing_text:
        print(trailing_text, end=" ", flush=True)

    total_elapsed = time.time() - start_time
    rtf = total_elapsed / max(0.01, duration_sec)

    print("\n\n" + "=" * 60)
    print("Final Transcript:")
    print(streamer.get_full_transcript())
    print(f"\nStats: Processed {duration_sec:.2f}s audio in {total_elapsed:.2f}s (RTF: {rtf:.2f}x)")
    print("=" * 60 + "\n")


def list_audio_devices():
    """Print all available audio input devices."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print("\nAvailable Audio Input Devices:")
        print("-" * 50)
        found = False
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                print(f" [{idx}] {dev['name']} (Channels: {dev['max_input_channels']}, Default SR: {dev['default_samplerate']} Hz)")
                found = True
        if not found:
            print(" No audio input devices found.")
        print("-" * 50 + "\n")
    except ImportError:
        raise ImportError("Please install sounddevice to list devices: pip install sounddevice")
