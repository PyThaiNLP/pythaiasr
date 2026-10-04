# -*- coding: utf-8 -*-
"""NVIDIA Nemotron 3.5 Streaming ASR ONNX Speech Recognition for PyThaiASR.

Supports:
- Offline audio file & array transcription
- Cache-aware real-time streaming transcription (chunk-by-chunk)
- Automatic download of ONNX models to ~/pythaiasr-data/nemotron-3.5-asr-streaming-onnx-int4/
- Multilingual prompt conditioning (default: Thai 'th-TH' / prompt ID 32)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None

from pythaiasr.download import (
    get_nemotron_asr_model_files,
    get_typhoon_nemotron_asr_model_files,
)
from pythaiasr.typhoon import (
    compute_stft_numpy,
    create_mel_filterbank,
    load_audio,
    resample_audio,
)

# Standard prompt dictionary for Nemotron 3.5 ASR
NEMOTRON_PROMPTS: Dict[str, int] = {
    "auto": 101,
    "en": 0,
    "en-US": 0,
    "en-GB": 1,
    "enGB": 1,
    "es": 3,
    "es-ES": 2,
    "es-US": 3,
    "zh": 4,
    "zh-CN": 4,
    "zh-TW": 5,
    "hi": 6,
    "hi-IN": 6,
    "ar": 7,
    "ar-AR": 7,
    "fr": 8,
    "fr-FR": 8,
    "fr-CA": 100,
    "de": 9,
    "de-DE": 9,
    "ja": 10,
    "ja-JP": 10,
    "ru": 11,
    "ru-RU": 11,
    "pt": 13,
    "pt-BR": 12,
    "pt-PT": 13,
    "ko": 14,
    "ko-KR": 14,
    "it": 15,
    "it-IT": 15,
    "nl": 16,
    "nl-NL": 16,
    "pl": 17,
    "pl-PL": 17,
    "tr": 18,
    "tr-TR": 18,
    "uk": 19,
    "uk-UA": 19,
    "ro": 20,
    "ro-RO": 20,
    "el": 21,
    "el-GR": 21,
    "cs": 22,
    "cs-CZ": 22,
    "hu": 23,
    "hu-HU": 23,
    "sv": 24,
    "sv-SE": 24,
    "da": 25,
    "da-DK": 25,
    "fi": 26,
    "fi-FI": 26,
    "no": 27,
    "no-NO": 27,
    "sk": 28,
    "sk-SK": 28,
    "hr": 29,
    "hr-HR": 29,
    "bg": 30,
    "bg-BG": 30,
    "lt": 31,
    "lt-LT": 31,
    "th": 32,
    "th-TH": 32,
    "thai": 32,
    "vi": 33,
    "vi-VN": 33,
    "id": 34,
    "id-ID": 34,
    "ms": 35,
    "ms-MY": 35,
}


def extract_nemotron_features(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 512,
    win_length: int = 400,
    hop_length: int = 160,
    preemph: float = 0.97,
    log_zero_guard: float = 5.96046448e-08,
) -> np.ndarray:
    """
    Extract 128-channel log-mel spectrogram features matching Nemotron 3.5 ASR.
    
    :param audio: 1D numpy array of audio samples (float32).
    :param sample_rate: Sample rate (must be 16000).
    :return: 2D numpy array of shape (num_frames, 128).
    """
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) == 0:
        return np.zeros((0, n_mels), dtype=np.float32)

    # 1. Pre-emphasis filter
    if preemph > 0:
        audio = np.append(audio[0], audio[1:] - preemph * audio[:-1])

    # 2. Compute STFT power spectrum
    power_spec = compute_stft_numpy(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # 3. Apply Slaney Mel filterbank
    fb = create_mel_filterbank(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sample_rate / 2.0)
    mel_spec = np.matmul(fb, power_spec)

    # 4. Log compression with zero guard
    log_mel = np.log(mel_spec + log_zero_guard).T

    return log_mel.astype(np.float32)


class NemotronStreamingASR:
    """Inference engine for Nemotron 3.5 Streaming ASR ONNX models."""

    def __init__(
        self,
        encoder_path: Optional[Union[str, Path]] = None,
        decoder_path: Optional[Union[str, Path]] = None,
        joint_path: Optional[Union[str, Path]] = None,
        vocab_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        precision: str = "int4",
        device: str = "auto",
        default_language: str = "th-TH",
        max_symbols_per_step: int = 10,
        model_dir: Optional[str] = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for Nemotron ASR. Please install it using:\n"
                "  pip install onnxruntime        # For CPU / Apple Silicon\n"
                "  pip install onnxruntime-gpu    # For NVIDIA CUDA"
            )

        # Download or locate model files if not specified
        if encoder_path is None or decoder_path is None or joint_path is None or vocab_path is None:
            def_enc, def_dec, def_joint, def_voc, def_cfg = get_nemotron_asr_model_files(
                model_dir=model_dir, precision=precision
            )
            encoder_path = encoder_path or def_enc
            decoder_path = decoder_path or def_dec
            joint_path = joint_path or def_joint
            vocab_path = vocab_path or def_voc
            config_path = config_path or def_cfg

        self.encoder_path = Path(encoder_path)
        self.decoder_path = Path(decoder_path)
        self.joint_path = Path(joint_path)
        self.vocab_path = Path(vocab_path)
        self.config_path = Path(config_path) if config_path else None
        self.precision = precision
        self.default_language = default_language
        self.max_symbols_per_step = max_symbols_per_step

        if not self.encoder_path.exists():
            raise FileNotFoundError(f"Encoder model not found: {self.encoder_path}")
        if not self.decoder_path.exists():
            raise FileNotFoundError(f"Decoder model not found: {self.decoder_path}")
        if not self.joint_path.exists():
            raise FileNotFoundError(f"Joint model not found: {self.joint_path}")
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")

        # Load vocabulary
        if str(self.vocab_path).endswith(".json"):
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                self.vocab: List[str] = json.load(f)
        else:
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                raw_lines = [line.strip("\r\n") for line in f.readlines()]
            self.vocab = []
            for line in raw_lines:
                if "\t" in line:
                    _, _, tok = line.partition("\t")
                    self.vocab.append(tok)
                else:
                    self.vocab.append(line)

        self.blank_id = len(self.vocab) - 1 if self.vocab else 15135
        self.hidden_dim = 640
        self.encoder_dim = 1024
        self.num_layers = 24
        self.left_context = 56
        self.conv_context = 8
        self.pre_encode_cache_size = 9
        self.chunk_frames = 56
        self.chunk_samples = 8960
        self.sample_rate = 16000

        # Load prompt dictionary if available from config
        self.prompt_dict = dict(NEMOTRON_PROMPTS)
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                    if "blank_token_id" in cfg_data:
                        self.blank_id = int(cfg_data["blank_token_id"])
                    elif "model" in cfg_data and "blank_token_id" in cfg_data["model"]:
                        self.blank_id = int(cfg_data["model"]["blank_token_id"])
                    prompts = cfg_data.get("prompt_dictionary") or cfg_data.get("model", {}).get("prompt_dictionary")
                    if isinstance(prompts, dict):
                        self.prompt_dict.update(prompts)
            except Exception:
                pass

        providers = self._resolve_providers(device, ort)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.encoder_session = ort.InferenceSession(str(self.encoder_path), opts, providers=providers)
        self.decoder_session = ort.InferenceSession(str(self.decoder_path), opts, providers=providers)
        self.joint_session = ort.InferenceSession(str(self.joint_path), opts, providers=providers)

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

    def resolve_lang_id(self, language: Optional[Union[str, int]] = None) -> int:
        """Resolve a language string or integer ID into the prompt token ID."""
        if language is None:
            language = self.default_language
        if isinstance(language, int):
            return language
        lang_str = str(language).strip()
        if lang_str in self.prompt_dict:
            return self.prompt_dict[lang_str]
        lower_lang = lang_str.lower()
        for k, v in self.prompt_dict.items():
            if k.lower() == lower_lang:
                return v
        return 32  # Default to Thai 'th-TH'

    @property
    def _is_modern_onnx(self) -> bool:
        if hasattr(self, "encoder_session") and hasattr(self.encoder_session, "get_inputs"):
            try:
                return any(i.name == "input_features" for i in self.encoder_session.get_inputs())
            except Exception:
                pass
        return False

    def init_encoder_cache(
        self, batch_size: int = 1
    ) -> Any:
        """Initialize streaming encoder caches."""
        if self._is_modern_onnx:
            caches = {}
            for inp in self.encoder_session.get_inputs():
                if inp.name not in ["input_features", "prompt_ids", "cache_mask", "is_first_chunk"]:
                    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                    caches[inp.name] = np.zeros(shape, dtype=np.float32)
            return caches
        cache_channel = np.zeros(
            (batch_size, self.num_layers, self.left_context, self.encoder_dim), dtype=np.float32
        )
        cache_time = np.zeros(
            (batch_size, self.num_layers, self.encoder_dim, self.conv_context), dtype=np.float32
        )
        cache_channel_len = np.zeros((batch_size,), dtype=np.int64)
        return cache_channel, cache_time, cache_channel_len

    def init_decoder_state(
        self, batch_size: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Initialize autoregressive decoder state."""
        h = np.zeros((2, batch_size, self.hidden_dim), dtype=np.float32)
        c = np.zeros((2, batch_size, self.hidden_dim), dtype=np.float32)
        last_token = self.blank_id
        return h, c, last_token

    def _prime_decoder(self, h: np.ndarray, c: np.ndarray, last_token: int):
        """Prime decoder state with initial token."""
        if hasattr(self, "decoder_session") and hasattr(self.decoder_session, "get_inputs"):
            try:
                inp_names = [i.name for i in self.decoder_session.get_inputs()]
                if "token" in inp_names:
                    return self.decoder_session.run(
                        None, {"token": np.array([[last_token]], dtype=np.int64), "h_in": h, "c_in": c}
                    )
            except Exception:
                pass
        return None, h, c

    def decode_token_ids(self, token_ids: List[int]) -> str:
        """Decode a sequence of vocabulary token IDs into readable text."""
        pieces = []
        for tid in token_ids:
            if 0 <= tid < len(self.vocab):
                token_str = self.vocab[tid]
                if token_str in {"<unk>", "<s>", "</s>", "<pad>", "<bos>", "<eos>", "<blank>"}:
                    continue
                # Skip language tags e.g. <th-TH>, <en-US>
                if token_str.startswith("<") and token_str.endswith(">"):
                    continue
                pieces.append(token_str)
        text = "".join(pieces)
        text = text.replace("\u2581", " ").replace(" ", " ").strip()
        return text

    def tokens_to_timestamp_chunks(
        self,
        token_ids: List[int],
        timestamps: List[int],
        sample_rate: int = 16000,
        mode: Union[bool, str] = True,
    ) -> List[Dict[str, Union[str, float, Tuple[float, float]]]]:
        """
        Group emitted token IDs into timestamped segments (each encoder frame is 80 ms).
        """
        if not token_ids or not timestamps or len(token_ids) != len(timestamps):
            return []

        frame_duration = 0.08  # 80ms per frame
        chunks = []

        is_char_mode = mode == "char"

        current_tokens: List[int] = []
        current_start_frame: Optional[int] = None
        current_end_frame: Optional[int] = None

        for tid, t in zip(token_ids, timestamps):
            if tid < 0 or tid >= len(self.vocab):
                continue
            token_str = self.vocab[tid]
            if token_str in {"<unk>", "<s>", "</s>", "<pad>", "<bos>", "<eos>", "<blank>"}:
                continue
            if token_str.startswith("<") and token_str.endswith(">"):
                continue

            is_word_boundary = token_str.startswith("\u2581") or token_str.startswith(" ")

            if is_char_mode:
                clean_text = token_str.replace("\u2581", " ").replace(" ", " ").strip()
                if clean_text:
                    s_time = round(t * frame_duration, 2)
                    e_time = round((t + 1) * frame_duration, 2)
                    chunks.append({
                        "text": clean_text,
                        "timestamp": (s_time, e_time),
                        "start": s_time,
                        "end": e_time,
                    })
                continue

            pause_boundary = (current_end_frame is not None) and ((t - current_end_frame) >= 6)

            if (is_word_boundary or pause_boundary) and current_tokens:
                text = self.decode_token_ids(current_tokens)
                if text:
                    s_time = round(current_start_frame * frame_duration, 2)
                    e_time = round((current_end_frame + 1) * frame_duration, 2)
                    chunks.append({
                        "text": text,
                        "timestamp": (s_time, e_time),
                        "start": s_time,
                        "end": e_time,
                    })
                current_tokens = []
                current_start_frame = None

            current_tokens.append(tid)
            if current_start_frame is None:
                current_start_frame = t
            current_end_frame = t

        if current_tokens and current_start_frame is not None and current_end_frame is not None:
            text = self.decode_token_ids(current_tokens)
            if text:
                s_time = round(current_start_frame * frame_duration, 2)
                e_time = round((current_end_frame + 1) * frame_duration, 2)
                chunks.append({
                    "text": text,
                    "timestamp": (s_time, e_time),
                    "start": s_time,
                    "end": e_time,
                })

        return chunks

    def greedy_decode(
        self,
        encoder_outputs: np.ndarray,
        h: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        last_token: Optional[int] = None,
    ) -> Tuple[List[int], List[int], np.ndarray, np.ndarray, int]:
        """
        Perform autoregressive RNN-T greedy decoding over encoder outputs.

        :param encoder_outputs: Array of shape (1, num_frames, 1024) or (1, num_frames, 640).
        :return: (emitted_tokens, token_timestamps, next_h, next_c, next_last_token)
        """
        if h is None or c is None or last_token is None:
            h, c, last_token = self.init_decoder_state(batch_size=encoder_outputs.shape[0])

        emitted_tokens: List[int] = []
        token_timestamps: List[int] = []
        num_frames = encoder_outputs.shape[1]

        # Check joint session inputs
        is_modern_joint = False
        if hasattr(self, "joint_session") and hasattr(self.joint_session, "get_inputs"):
            try:
                is_modern_joint = any(i.name == "encoder_frame" for i in self.joint_session.get_inputs())
            except Exception:
                pass

        if is_modern_joint:
            dec_out, h, c = self._prime_decoder(h, c, last_token)
            for t in range(num_frames):
                enc_frame = encoder_outputs[0, t : t + 1]  # (1, 640)
                symbols_added = 0
                while symbols_added < self.max_symbols_per_step:
                    joint_out = self.joint_session.run(
                        None, {"encoder_frame": enc_frame, "decoder_out": dec_out}
                    )[0]
                    predicted_token = int(np.argmax(joint_out, axis=-1)[0])
                    if predicted_token == self.blank_id:
                        break
                    emitted_tokens.append(predicted_token)
                    token_timestamps.append(t)
                    last_token = predicted_token
                    dec_out, h, c = self.decoder_session.run(
                        None,
                        {"token": np.array([[last_token]], dtype=np.int64), "h_in": h, "c_in": c},
                    )
                    symbols_added += 1
            return emitted_tokens, token_timestamps, h, c, last_token

        # Legacy fallback
        for t in range(num_frames):
            enc_frame = encoder_outputs[:, t : t + 1, :]
            symbols_added = 0

            while symbols_added < self.max_symbols_per_step:
                targets = np.array([[last_token]], dtype=np.int64)

                dec_out, next_h, next_c = self.decoder_session.run(
                    None,
                    {
                        "targets": targets,
                        "h_in": h,
                        "c_in": c,
                    },
                )

                if dec_out.ndim == 3 and dec_out.shape[1] == self.hidden_dim:
                    dec_out = np.transpose(dec_out, (0, 2, 1))

                joint_out = self.joint_session.run(
                    None,
                    {
                        "encoder_output": enc_frame,
                        "decoder_output": dec_out,
                    },
                )[0]

                logits = joint_out[0, 0, 0, :]
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

        return emitted_tokens, token_timestamps, h, c, last_token

    def transcribe(
        self,
        data: Union[str, Path, np.ndarray],
        sampling_rate: int = 16000,
        sample_rate: Optional[int] = None,
        return_timestamps: Optional[bool] = None,
        timestamps: Optional[bool] = None,
        language: Optional[Union[str, int]] = None,
    ) -> Union[str, Dict]:
        """
        Transcribe an audio file or numpy array using Nemotron 3.5 Streaming ASR.

        :param data: Audio file path or 1D numpy array.
        :param sampling_rate: Audio sampling rate (default 16000).
        :param sample_rate: Alias for sampling_rate.
        :param return_timestamps: Return timestamp chunks if True.
        :param timestamps: Alias for return_timestamps.
        :param language: Language code (e.g. 'th-TH', 'en-US') or prompt ID.
        :return: Transcribed string or dictionary with timestamps.
        """
        if sample_rate is not None:
            sampling_rate = sample_rate
        if isinstance(data, (str, Path)):
            audio = load_audio(data, target_sr=self.sample_rate)
        else:
            audio = np.asarray(data, dtype=np.float32).flatten()
            if sampling_rate != self.sample_rate:
                audio = resample_audio(audio, orig_sr=sampling_rate, target_sr=self.sample_rate)

        lang_id = self.resolve_lang_id(language)

        # Extract 128-mel log spectrogram
        mel_features = extract_nemotron_features(audio, sample_rate=self.sample_rate)
        total_mel_frames = len(mel_features)

        all_tokens: List[int] = []
        all_timestamps: List[int] = []
        global_frame_offset = 0

        if self._is_modern_onnx:
            caches = self.init_encoder_cache()
            cache_valid = 0
            left = self.left_context
            enc_frames_per_chunk = self.chunk_frames // 8  # 7 frames for 56 mel frames
            cache_carry = {
                o.name: o.name.replace("_out_", "_")
                for o in self.encoder_session.get_outputs()[1:]
            }
            h, c, last_token = self.init_decoder_state()
            dec_out, h, c = self._prime_decoder(h, c, last_token)

            for start_idx in range(0, total_mel_frames, self.chunk_frames):
                chunk = mel_features[start_idx : start_idx + self.chunk_frames]
                if len(chunk) < self.chunk_frames:
                    chunk = np.pad(chunk, ((0, self.chunk_frames - len(chunk)), (0, 0)), mode="constant")

                mask = np.zeros((1, 1, 1, left + enc_frames_per_chunk), dtype=np.float32)
                invalid = left - cache_valid
                if invalid > 0:
                    mask[..., :invalid] = -1e9

                feed = {
                    "input_features": chunk[None, ...].astype(np.float32),
                    "prompt_ids": np.array([lang_id], dtype=np.int64),
                    "cache_mask": mask,
                }
                feed.update(caches)
                outs = self.encoder_session.run(None, feed)
                for o, val in zip(self.encoder_session.get_outputs()[1:], outs[1:]):
                    caches[cache_carry[o.name]] = val
                cache_valid = min(left, cache_valid + enc_frames_per_chunk)

                enc_out = outs[0]
                for t in range(enc_frames_per_chunk):
                    enc_frame = enc_out[0, t : t + 1]
                    symbols = 0
                    while symbols < self.max_symbols_per_step:
                        logits = self.joint_session.run(
                            None, {"encoder_frame": enc_frame, "decoder_out": dec_out}
                        )[0]
                        token = int(np.argmax(logits, axis=-1)[0])
                        if token == self.blank_id:
                            break
                        all_tokens.append(token)
                        all_timestamps.append(global_frame_offset + t)
                        last_token = token
                        dec_out, h, c = self.decoder_session.run(
                            None,
                            {"token": np.array([[token]], dtype=np.int64), "h_in": h, "c_in": c},
                        )
                        symbols += 1

                global_frame_offset += enc_frames_per_chunk

            text = self.decode_token_ids(all_tokens)
            use_ts = return_timestamps or bool(timestamps)
            if use_ts:
                chunks = self.tokens_to_timestamp_chunks(all_tokens, all_timestamps, sample_rate=self.sample_rate)
                return {"text": text, "chunks": chunks, "timestamps": chunks}
            return text

        # Legacy fallback
        cache_channel, cache_time, cache_channel_len = self.init_encoder_cache()
        h, c, last_token = self.init_decoder_state()
        prev_mel = np.zeros((self.pre_encode_cache_size, 128), dtype=np.float32)

        for start_idx in range(0, total_mel_frames, self.chunk_frames):
            chunk = mel_features[start_idx : start_idx + self.chunk_frames]
            chunk_len = len(chunk)

            if chunk_len < self.chunk_frames:
                chunk = np.pad(chunk, ((0, self.chunk_frames - chunk_len), (0, 0)), mode="constant")

            chunk_65 = np.concatenate([prev_mel, chunk], axis=0)
            audio_signal = np.expand_dims(chunk_65, axis=0).astype(np.float32)

            enc_outs, enc_lens, cache_channel, cache_time, cache_channel_len = self.encoder_session.run(
                None,
                {
                    "audio_signal": audio_signal,
                    "length": np.array([65], dtype=np.int64),
                    "cache_last_channel": cache_channel,
                    "cache_last_time": cache_time,
                    "cache_last_channel_len": cache_channel_len,
                    "lang_id": np.array([lang_id], dtype=np.int64),
                },
            )

            prev_mel = chunk[-self.pre_encode_cache_size :]

            chunk_tokens, chunk_ts, h, c, last_token = self.greedy_decode(
                enc_outs, h=h, c=c, last_token=last_token
            )

            for tid, t in zip(chunk_tokens, chunk_ts):
                all_tokens.append(tid)
                all_timestamps.append(global_frame_offset + t)

            global_frame_offset += enc_outs.shape[1]

        text = self.decode_token_ids(all_tokens)
        use_ts = return_timestamps or bool(timestamps)

        if use_ts:
            chunks = self.tokens_to_timestamp_chunks(all_tokens, all_timestamps, sample_rate=self.sample_rate)
            return {
                "text": text,
                "chunks": chunks,
                "timestamps": chunks,
            }

        return text


class NemotronStreamASR:
    """Stateful chunk-by-chunk real-time transcriber for Nemotron 3.5 Streaming ASR."""

    def __init__(
        self,
        model: NemotronStreamingASR,
        sample_rate: int = 16000,
        step_sec: float = 0.56,
        language: Optional[Union[str, int]] = None,
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.step_sec = step_sec
        self.chunk_samples = int(sample_rate * step_sec)  # 8960 samples for 0.56s
        self.lang_id = model.resolve_lang_id(language)
        self.reset()

    def reset(self) -> None:
        """Reset internal streaming states."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.emitted_tokens: List[int] = []

        if self.model._is_modern_onnx:
            self.caches = self.model.init_encoder_cache()
            self.cache_valid = 0
            self.h, self.c, self.last_token = self.model.init_decoder_state()
            self.dec_out, self.h, self.c = self.model._prime_decoder(self.h, self.c, self.last_token)
            self.cache_carry = {
                o.name: o.name.replace("_out_", "_")
                for o in self.model.encoder_session.get_outputs()[1:]
            }
        else:
            self.prev_mel = np.zeros((self.model.pre_encode_cache_size, 128), dtype=np.float32)
            self.cache_channel, self.cache_time, self.cache_channel_len = self.model.init_encoder_cache()
            self.h, self.c, self.last_token = self.model.init_decoder_state()

    def process_chunk(self, chunk: np.ndarray) -> str:
        """Process incoming raw audio chunk and return newly emitted text."""
        chunk = np.asarray(chunk, dtype=np.float32).flatten()
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])

        new_tokens: List[int] = []

        while len(self.audio_buffer) >= self.chunk_samples:
            sub_audio = self.audio_buffer[: self.chunk_samples]
            self.audio_buffer = self.audio_buffer[self.chunk_samples :]

            mel = extract_nemotron_features(sub_audio, sample_rate=self.sample_rate)
            if len(mel) > self.model.chunk_frames:
                mel = mel[: self.model.chunk_frames]
            elif len(mel) < self.model.chunk_frames:
                mel = np.pad(mel, ((0, self.model.chunk_frames - len(mel)), (0, 0)), mode="constant")

            if self.model._is_modern_onnx:
                left = self.model.left_context
                enc_frames_per_chunk = self.model.chunk_frames // 8
                mask = np.zeros((1, 1, 1, left + enc_frames_per_chunk), dtype=np.float32)
                invalid = left - self.cache_valid
                if invalid > 0:
                    mask[..., :invalid] = -1e9

                feed = {
                    "input_features": mel[None, ...].astype(np.float32),
                    "prompt_ids": np.array([self.lang_id], dtype=np.int64),
                    "cache_mask": mask,
                }
                feed.update(self.caches)
                outs = self.model.encoder_session.run(None, feed)
                for o, val in zip(self.model.encoder_session.get_outputs()[1:], outs[1:]):
                    self.caches[self.cache_carry[o.name]] = val
                self.cache_valid = min(left, self.cache_valid + enc_frames_per_chunk)

                enc_out = outs[0]
                for t in range(enc_frames_per_chunk):
                    enc_frame = enc_out[0, t : t + 1]
                    symbols = 0
                    while symbols < self.model.max_symbols_per_step:
                        logits = self.model.joint_session.run(
                            None, {"encoder_frame": enc_frame, "decoder_out": self.dec_out}
                        )[0]
                        token = int(np.argmax(logits, axis=-1)[0])
                        if token == self.model.blank_id:
                            break
                        new_tokens.append(token)
                        self.emitted_tokens.append(token)
                        self.last_token = token
                        self.dec_out, self.h, self.c = self.model.decoder_session.run(
                            None,
                            {"token": np.array([[token]], dtype=np.int64), "h_in": self.h, "c_in": self.c},
                        )
                        symbols += 1
            else:
                chunk_65 = np.concatenate([self.prev_mel, mel], axis=0)
                self.prev_mel = mel[-self.model.pre_encode_cache_size :]
                audio_signal = np.expand_dims(chunk_65, axis=0).astype(np.float32)

                enc_outs, _, self.cache_channel, self.cache_time, self.cache_channel_len = (
                    self.model.encoder_session.run(
                        None,
                        {
                            "audio_signal": audio_signal,
                            "length": np.array([65], dtype=np.int64),
                            "cache_last_channel": self.cache_channel,
                            "cache_last_time": self.cache_time,
                            "cache_last_channel_len": self.cache_channel_len,
                            "lang_id": np.array([self.lang_id], dtype=np.int64),
                        },
                    )
                )

                toks, _, self.h, self.c, self.last_token = self.model.greedy_decode(
                    enc_outs, h=self.h, c=self.c, last_token=self.last_token
                )
                new_tokens.extend(toks)
                self.emitted_tokens.extend(toks)

        if new_tokens:
            return self.model.decode_token_ids(new_tokens)
        return ""

    def flush(self) -> str:
        """Process any remaining audio in the buffer and return trailing text."""
        if len(self.audio_buffer) == 0:
            return ""

        pad_len = self.chunk_samples - len(self.audio_buffer)
        sub_audio = np.pad(self.audio_buffer, (0, pad_len), mode="constant")
        self.audio_buffer = np.array([], dtype=np.float32)

        mel = extract_nemotron_features(sub_audio, sample_rate=self.sample_rate)
        if len(mel) > self.model.chunk_frames:
            mel = mel[: self.model.chunk_frames]
        elif len(mel) < self.model.chunk_frames:
            mel = np.pad(mel, ((0, self.model.chunk_frames - len(mel)), (0, 0)), mode="constant")

        new_tokens = []
        if self.model._is_modern_onnx:
            left = self.model.left_context
            enc_frames_per_chunk = self.model.chunk_frames // 8
            mask = np.zeros((1, 1, 1, left + enc_frames_per_chunk), dtype=np.float32)
            invalid = left - self.cache_valid
            if invalid > 0:
                mask[..., :invalid] = -1e9

            feed = {
                "input_features": mel[None, ...].astype(np.float32),
                "prompt_ids": np.array([self.lang_id], dtype=np.int64),
                "cache_mask": mask,
            }
            feed.update(self.caches)
            outs = self.model.encoder_session.run(None, feed)
            for o, val in zip(self.model.encoder_session.get_outputs()[1:], outs[1:]):
                self.caches[self.cache_carry[o.name]] = val
            self.cache_valid = min(left, self.cache_valid + enc_frames_per_chunk)

            enc_out = outs[0]
            for t in range(enc_frames_per_chunk):
                enc_frame = enc_out[0, t : t + 1]
                symbols = 0
                while symbols < self.model.max_symbols_per_step:
                    logits = self.model.joint_session.run(
                        None, {"encoder_frame": enc_frame, "decoder_out": self.dec_out}
                    )[0]
                    token = int(np.argmax(logits, axis=-1)[0])
                    if token == self.model.blank_id:
                        break
                    new_tokens.append(token)
                    self.emitted_tokens.append(token)
                    self.last_token = token
                    self.dec_out, self.h, self.c = self.model.decoder_session.run(
                        None,
                        {"token": np.array([[token]], dtype=np.int64), "h_in": self.h, "c_in": self.c},
                    )
                    symbols += 1
        else:
            chunk_65 = np.concatenate([self.prev_mel, mel], axis=0)
            audio_signal = np.expand_dims(chunk_65, axis=0).astype(np.float32)

            enc_outs, _, self.cache_channel, self.cache_time, self.cache_channel_len = (
                self.model.encoder_session.run(
                    None,
                    {
                        "audio_signal": audio_signal,
                        "length": np.array([65], dtype=np.int64),
                        "cache_last_channel": self.cache_channel,
                        "cache_last_time": self.cache_time,
                        "cache_last_channel_len": self.cache_channel_len,
                        "lang_id": np.array([self.lang_id], dtype=np.int64),
                    },
                )
            )

            toks, _, self.h, self.c, self.last_token = self.model.greedy_decode(
                enc_outs, h=self.h, c=self.c, last_token=self.last_token
            )
            new_tokens.extend(toks)
            self.emitted_tokens.extend(toks)

        if new_tokens:
            return self.model.decode_token_ids(new_tokens)
        return ""

    def get_full_transcript(self) -> str:
        """Get the complete decoded transcript of all tokens received so far."""
        return self.model.decode_token_ids(self.emitted_tokens)


# Aliases
TyphoonNemotronStreamingASR = NemotronStreamingASR
TyphoonNemotronASR = NemotronStreamingASR
TyphoonNemotronStreamASR = NemotronStreamASR
RealtimeStreamTyphoonNemotron = NemotronStreamASR
extract_typhoon_nemotron_features = extract_nemotron_features

NemotronASR = NemotronStreamingASR
NemotronStreamingAudioRecognizer = NemotronStreamASR
RealtimeStreamNemotron = NemotronStreamASR
