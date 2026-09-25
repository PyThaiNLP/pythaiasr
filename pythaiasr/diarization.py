# -*- coding: utf-8 -*-
"""Speech Diarization with ONNX for PyThaiASR.

Provides:
- Diarization: ONNX-based speaker diarization engine using Pyannote Segmentation 3.0.
- diarize: Public function to extract timestamped speaker segments.
- asr_diarize: Public function to extract speaker segments and transcribe them with ASR.
"""

from __future__ import annotations

import math
import os
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from pythaiasr.download import (
    get_diarization_model_files,
    get_nemotron_diarization_model_files,
)
from pythaiasr.typhoon import load_audio, resample_audio


# Constants for Pyannote Segmentation 3.0
_SAMPLE_RATE = 16000
_WINDOW_SECONDS = 10.0
_WINDOW_SAMPLES = int(_SAMPLE_RATE * _WINDOW_SECONDS)  # 160,000 samples
_STEP_SAMPLES = 270
_OFFSET_SAMPLES = 990


def _powerset_to_multilabel(logits: np.ndarray) -> np.ndarray:
    """
    Convert Pyannote Segmentation powerset logits to marginal speaker probabilities.

    The 7 powerset classes correspond to:
        0: non-speech
        1: speaker 1
        2: speaker 2
        3: speaker 3
        4: speakers 1 and 2
        5: speakers 1 and 3
        6: speakers 2 and 3

    Marginal probabilities:
        P(Spk1) = P(1) + P(4) + P(5)
        P(Spk2) = P(2) + P(4) + P(6)
        P(Spk3) = P(3) + P(5) + P(6)

    :param np.ndarray logits: Array of shape (..., num_frames, 7)
    :return: Array of marginal probabilities of shape (..., num_frames, 3)
    """
    # Numerically stable softmax along last axis
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    p1 = probs[..., 1] + probs[..., 4] + probs[..., 5]
    p2 = probs[..., 2] + probs[..., 4] + probs[..., 6]
    p3 = probs[..., 3] + probs[..., 5] + probs[..., 6]

    return np.stack([p1, p2, p3], axis=-1)


def _binarize_timeline(
    speaker_probs: np.ndarray,
    total_samples: int,
    sample_rate: int = _SAMPLE_RATE,
    onset: float = 0.5,
    offset: float = 0.5,
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.5,
) -> List[Dict[str, Union[float, str]]]:
    """
    Convert frame-level speaker probabilities to discrete timestamped segments.

    Uses hysteresis thresholding with onset and offset, filters out short turns,
    and merges brief pauses.
    """
    total_duration = total_samples / sample_rate
    num_frames, num_speakers = speaker_probs.shape
    segments = []

    for spk_idx in range(num_speakers):
        probs = speaker_probs[:, spk_idx]
        is_active = False
        start_frame = 0
        spk_segments = []

        for f in range(num_frames):
            p = probs[f]
            if not is_active:
                if p >= onset:
                    is_active = True
                    start_frame = f
            else:
                if p < offset:
                    is_active = False
                    end_frame = f
                    # Convert frames to seconds
                    start_t = max(0.0, (start_frame * _STEP_SAMPLES) / sample_rate)
                    end_t = min(total_duration, (end_frame * _STEP_SAMPLES + _OFFSET_SAMPLES) / sample_rate)
                    if end_t > start_t:
                        spk_segments.append((start_t, end_t))

        if is_active:
            start_t = max(0.0, (start_frame * _STEP_SAMPLES) / sample_rate)
            end_t = total_duration
            if end_t > start_t:
                spk_segments.append((start_t, end_t))

        # Filter out turns shorter than min_duration_on
        filtered = [seg for seg in spk_segments if (seg[1] - seg[0]) >= min_duration_on]
        if not filtered:
            continue

        # Merge gaps smaller than min_duration_off
        merged = [filtered[0]]
        for curr_s, curr_e in filtered[1:]:
            prev_s, prev_e = merged[-1]
            if (curr_s - prev_e) < min_duration_off:
                merged[-1] = (prev_s, max(prev_e, curr_e))
            else:
                merged.append((curr_s, curr_e))

        for s, e in merged:
            segments.append({
                "start": round(s, 3),
                "end": round(e, 3),
                "speaker": f"SPEAKER_{spk_idx:02d}",
            })

    # Sort all segments by start time
    segments.sort(key=lambda x: (x["start"], x["end"]))
    return segments


# Constants for Nemotron-3 Diarization (joosthel/Nemotron-3-Diarization-ONNX)
NEMOTRON_SAMPLE_RATE = 16000
NEMOTRON_HOP_LENGTH = 160
NEMOTRON_N_FFT = 512
NEMOTRON_PREEMPHASIS = 0.97
NEMOTRON_FRAME_DURATION = NEMOTRON_HOP_LENGTH / NEMOTRON_SAMPLE_RATE  # 0.01s (10ms)
LOG_MEL_WINDOW_FRAMES = 6000

NEMOTRON_DIARIZATION_MODELS = [
    "nemotron",
    "nemotron_diarization",
    "nemotron-diarization",
    "nemotron-3-diarization",
    "nemotron-3-diarization-onnx",
    "joosthel/Nemotron-3-Diarization-ONNX",
    "joosthel/nemotron-3-diarization-onnx",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def _stable_topk_indices(scores: np.ndarray, k: int, axis: int) -> np.ndarray:
    """Indices of the k largest entries along axis, ties broken toward the lower index."""
    order = np.argsort(-scores, axis=axis, kind="stable")
    return np.take(order, np.arange(k), axis=axis)


class NumpySpeakerCache:
    """
    Line-by-line numpy port of Nemotron3DiarizationSpeakerCache.
    Eagerly allocates fixed-size embeds/probs/fifo buffers.
    """

    def __init__(
        self,
        constants: dict,
        fifo_length: int,
        speaker_cache_update_period: int,
        topk_fn=_stable_topk_indices,
    ):
        self._topk_fn = topk_fn
        self.hidden_size = int(constants["hidden_size"])
        self.num_speakers = int(constants["num_speakers"])
        self.subsampling_factor = int(constants["subsampling_factor"])
        self.speaker_cache_length = int(constants["speaker_cache_length"])
        self.num_silence_frames = int(constants["speaker_cache_silence_frames_per_speaker"])
        self.prediction_score_threshold = float(constants["prediction_score_threshold"])
        self.latest_frames_score_boost = float(constants["latest_frames_score_boost"])
        self.silence_embeds = constants["silence_embeds"].astype(np.float32)

        self.fifo_length = fifo_length
        self.speaker_cache_update_period = speaker_cache_update_period

        budget = self.speaker_cache_length // self.num_speakers - self.num_silence_frames
        self.min_positive_scores = math.floor(budget * float(constants["min_positive_scores_rate"]))
        self.num_strong_boosted_frames = math.floor(budget * float(constants["strong_boost_rate"]))
        self.num_weak_boosted_frames = math.floor(budget * float(constants["weak_boost_rate"]))

        self.embeds = np.zeros((1, self.speaker_cache_length, self.hidden_size), dtype=np.float32)
        self.probs = np.zeros((1, self.speaker_cache_length, self.num_speakers), dtype=np.float32)
        self.fifo = np.zeros((1, self.fifo_length, self.hidden_size), dtype=np.float32)
        self.num_cache_frames = 0
        self.num_fifo_frames = 0
        self.is_compressed = False

    def get_embeds(self) -> np.ndarray:
        return np.concatenate(
            [self.embeds[:, : self.num_cache_frames], self.fifo[:, : self.num_fifo_frames]],
            axis=1,
        )

    def _pool_probs(self, logits: np.ndarray) -> np.ndarray:
        """Speaker probabilities at encoder frame rate (avg_pool1d(kernel=stride=8))."""
        factor = self.subsampling_factor
        probs = _sigmoid(logits)
        batch, num_frames, num_speakers = probs.shape
        pooled = probs.reshape(batch, num_frames // factor, factor, num_speakers).mean(axis=2)
        return pooled.astype(self.probs.dtype)

    def _num_popped_frames(self, num_fifo_frames: int) -> int:
        if num_fifo_frames <= self.fifo_length:
            return 0
        num_popped = max(self.speaker_cache_update_period, num_fifo_frames - self.fifo_length)
        return min(num_popped, num_fifo_frames)

    def update(self, chunk_input_embeds: np.ndarray, chunk_logits: np.ndarray, num_chunk_frames: int):
        num_cache_frames, num_fifo_frames = self.num_cache_frames, self.num_fifo_frames
        probs = self._pool_probs(chunk_logits)

        chunk_start = num_cache_frames + num_fifo_frames
        chunk_embeds = chunk_input_embeds[:, chunk_start : chunk_start + num_chunk_frames]
        fifo_embeds = np.concatenate([self.fifo[:, :num_fifo_frames], chunk_embeds], axis=1)

        num_popped = self._num_popped_frames(fifo_embeds.shape[1])
        if num_popped:
            fifo_probs = probs[:, num_cache_frames : num_cache_frames + fifo_embeds.shape[1]]
            stored_probs = self.probs[:, :num_cache_frames] if self.is_compressed else probs[:, :num_cache_frames]
            cache_embeds = np.concatenate([self.embeds[:, :num_cache_frames], fifo_embeds[:, :num_popped]], axis=1)
            cache_probs = np.concatenate([stored_probs, fifo_probs[:, :num_popped]], axis=1)
            fifo_embeds = fifo_embeds[:, num_popped:]

            if cache_embeds.shape[1] > self.speaker_cache_length:
                cache_embeds, cache_probs = self._compress(cache_embeds, cache_probs)
                self.is_compressed = True
            self.num_cache_frames = cache_embeds.shape[1]

            self.embeds[:, : self.num_cache_frames] = cache_embeds
            self.probs[:, : self.num_cache_frames] = cache_probs

        self.num_fifo_frames = fifo_embeds.shape[1]
        self.fifo[:, : self.num_fifo_frames] = fifo_embeds

    def _get_frame_scores(self, probs: np.ndarray) -> np.ndarray:
        threshold = self.prediction_score_threshold
        log_probs = np.log(np.clip(probs, threshold, None))
        log_complements = np.log(np.clip(1.0 - probs, threshold, None))

        scores = log_probs - log_complements + log_complements.sum(axis=-1, keepdims=True) - math.log(0.5)

        is_speech = probs > 0.5
        scores = np.where(is_speech, scores, -np.inf)
        is_positive = scores > 0
        has_enough_positive = is_positive.sum(axis=1, keepdims=True) >= self.min_positive_scores
        scores = np.where(~is_positive & is_speech & has_enough_positive, -np.inf, scores)
        return scores

    def _boost_scores(self, scores: np.ndarray, num_boosted: int, boost: float) -> np.ndarray:
        if num_boosted <= 0:
            return scores
        topk_idx = self._topk_fn(scores, num_boosted, axis=1)
        scores = scores.copy()
        boosted = np.take_along_axis(scores, topk_idx, axis=1) + boost
        np.put_along_axis(scores, topk_idx, boosted, axis=1)
        return scores

    def _compress(self, embeds: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size, num_frames, num_speakers = probs.shape

        scores = self._get_frame_scores(probs)
        scores = scores.copy()
        scores[:, self.speaker_cache_length :] += self.latest_frames_score_boost

        scores = self._boost_scores(scores, self.num_strong_boosted_frames, boost=-2.0 * math.log(0.5))
        scores = self._boost_scores(scores, self.num_weak_boosted_frames, boost=-math.log(0.5))
        scores = np.pad(scores, ((0, 0), (0, self.num_silence_frames), (0, 0)), constant_values=np.inf)
        embeds = np.concatenate(
            [embeds, np.broadcast_to(self.silence_embeds.reshape(1, 1, -1), (batch_size, 1, embeds.shape[-1]))],
            axis=1,
        )
        probs = np.pad(probs, ((0, 0), (0, 1), (0, 0)))

        num_scored_frames = num_frames + self.num_silence_frames
        sentinel = num_scored_frames * num_speakers
        flat_scores = scores.transpose(0, 2, 1).reshape(batch_size, -1)

        topk_idx = self._topk_fn(flat_scores, self.speaker_cache_length, axis=1)
        topk_scores = np.take_along_axis(flat_scores, topk_idx, axis=1)
        topk_idx = np.where(topk_scores == -np.inf, sentinel, topk_idx)
        topk_idx = np.sort(topk_idx, axis=1)
        frame_indices = np.where(topk_idx == sentinel, num_frames, np.minimum(topk_idx % num_scored_frames, num_frames))

        batch_indices = np.arange(batch_size)[:, None]
        return embeds[batch_indices, frame_indices], probs[batch_indices, frame_indices]


class NemotronDiarization:
    """
    ONNX-based Speaker Diarization using NVIDIA Nemotron-3 Diarization (joosthel/Nemotron-3-Diarization-ONNX).

    Pure-numpy + onnxruntime offline speaker diarization engine, supporting up to 8
    concurrently tracked speakers with 10ms frame resolution.
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        precision: str = "int8",
        device: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        if ort is None:
            raise ImportError(
                "onnxruntime is required for Nemotron Diarization. "
                "Install it with: pip install onnxruntime"
            )

        self.device = device or "auto"
        self.precision = precision

        prep_path, model_path, const_path = get_nemotron_diarization_model_files(
            model_dir=model_dir, precision=precision
        )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3
        if threads:
            sess_options.intra_op_num_threads = threads

        if providers is None:
            providers = self._resolve_providers(self.device, ort)

        self.preprocessor_core = ort.InferenceSession(prep_path, sess_options=sess_options, providers=providers)
        self.model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        self.constants = dict(np.load(const_path))

        self.subsampling_factor = int(self.constants["subsampling_factor"])
        self.chunk_length = int(self.constants["chunk_length"])
        self.chunk_right_context = int(self.constants["chunk_right_context"])
        self.fifo_length = int(self.constants["fifo_length"])
        self.speaker_cache_update_period = int(self.constants["speaker_cache_update_period"])

    @staticmethod
    def _resolve_providers(device: str, ort_mod) -> List[str]:
        avail = ort_mod.get_available_providers()
        if device in ("cuda", "gpu"):
            if "CUDAExecutionProvider" in avail:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            raise RuntimeError("CUDA requested, but 'CUDAExecutionProvider' is not available in onnxruntime.")
        elif device == "cpu":
            return ["CPUExecutionProvider"]
        else:  # auto
            if "CUDAExecutionProvider" in avail:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]

    def log_mel(self, waveform: np.ndarray) -> np.ndarray:
        """Bounded-memory windowed log-mel feature extraction."""
        waveform = np.asarray(waveform, dtype=np.float32)
        num_samples = waveform.shape[0]
        num_frames_total = 1 + num_samples // NEMOTRON_HOP_LENGTH
        mel = np.empty((1, num_frames_total, 128), dtype=np.float32)

        start_frame = 0
        while start_frame < num_frames_total:
            end_frame = min(start_frame + LOG_MEL_WINDOW_FRAMES, num_frames_total)
            raw_start = start_frame * NEMOTRON_HOP_LENGTH - NEMOTRON_N_FFT // 2
            raw_end = (end_frame - 1) * NEMOTRON_HOP_LENGTH - NEMOTRON_N_FFT // 2 + NEMOTRON_N_FFT
            lookback = raw_start - 1
            real_start = max(0, lookback)
            real_end = max(real_start, min(num_samples, raw_end))
            real_segment = waveform[real_start:real_end]

            if lookback >= 0:
                preemph_real = real_segment[1:] - NEMOTRON_PREEMPHASIS * real_segment[:-1]
                content_start = raw_start
            else:
                preemph_real = real_segment.copy()
                preemph_real[1:] -= NEMOTRON_PREEMPHASIS * real_segment[:-1]
                content_start = real_start

            left_pad = content_start - raw_start
            right_pad = raw_end - real_end
            preemphasized = (
                np.pad(preemph_real, (left_pad, right_pad))
                if (left_pad or right_pad)
                else preemph_real
            )

            (mel_chunk,) = self.preprocessor_core.run(None, {"preemphasized": preemphasized[None, :]})
            mel[:, start_frame:end_frame] = mel_chunk[:, : end_frame - start_frame]
            start_frame = end_frame

        if num_frames_total:
            mel[:, -1, :] = 0.0
        return mel

    def run_chunk(
        self, chunk_mel: np.ndarray, chunk_mel_length: int, context_embeds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        logits, embeds = self.model.run(
            None,
            {
                "chunk_mel": chunk_mel,
                "chunk_mel_length": np.array(chunk_mel_length, dtype=np.int64),
                "context_embeds": context_embeds,
                "context_length": np.array(context_embeds.shape[1], dtype=np.int64),
            },
        )
        return logits, embeds

    def predict_proba(
        self,
        data: Union[str, Path, np.ndarray],
        sampling_rate: int = NEMOTRON_SAMPLE_RATE,
    ) -> np.ndarray:
        """
        Get frame-level speaker probabilities of shape (1, num_frames, 8).
        """
        if isinstance(data, (str, Path)):
            audio = load_audio(data, target_sr=NEMOTRON_SAMPLE_RATE)
        elif isinstance(data, np.ndarray):
            audio = np.asarray(data, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sampling_rate != NEMOTRON_SAMPLE_RATE:
                audio = resample_audio(audio, orig_sr=sampling_rate, target_sr=NEMOTRON_SAMPLE_RATE)
        else:
            raise TypeError(f"Unsupported data type for diarization: {type(data)}")

        if len(audio) == 0:
            return np.zeros((1, 0, 8), dtype=np.float32)

        mel = self.log_mel(audio)
        num_mel_frames = mel.shape[1]
        num_embeds = -(-num_mel_frames // self.subsampling_factor)
        valid_mel_frames = max(num_mel_frames - 1, 0)

        cache = NumpySpeakerCache(
            self.constants,
            fifo_length=self.fifo_length,
            speaker_cache_update_period=self.speaker_cache_update_period,
            topk_fn=_stable_topk_indices,
        )

        chunk_logits = []
        for start_idx in range(0, num_embeds, self.chunk_length):
            end_idx = min(start_idx + self.chunk_length, num_embeds)
            num_chunk_frames = end_idx - start_idx

            mel_start = start_idx * self.subsampling_factor
            mel_end = min((end_idx + self.chunk_right_context) * self.subsampling_factor, num_mel_frames)
            chunk_mel = mel[:, mel_start:mel_end]
            chunk_mel_length = max(min(mel_end, valid_mel_frames) - mel_start, 0)

            context_embeds = cache.get_embeds()
            context_length = context_embeds.shape[1]

            logits, chunk_input_embeds = self.run_chunk(chunk_mel, chunk_mel_length, context_embeds)
            cache.update(chunk_input_embeds, logits, num_chunk_frames)

            start_logit_idx = context_length * self.subsampling_factor
            end_logit_idx = (context_length + num_chunk_frames) * self.subsampling_factor
            chunk_logits.append(logits[:, start_logit_idx:end_logit_idx])

        if not chunk_logits:
            return np.zeros((1, 0, 8), dtype=np.float32)

        logits = np.concatenate(chunk_logits, axis=1)[:, :num_mel_frames]
        return _sigmoid(logits)

    def diarize(
        self,
        data: Union[str, Path, np.ndarray],
        sampling_rate: int = NEMOTRON_SAMPLE_RATE,
        threshold: float = 0.5,
        onset: Optional[float] = None,
        offset: Optional[float] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        min_duration_on: float = 0.3,
        min_duration_off: float = 0.5,
        **kwargs,
    ) -> List[Dict[str, Union[float, str]]]:
        """
        Diarize audio data and return list of speaker turns.
        """
        probs = self.predict_proba(data, sampling_rate=sampling_rate)
        if probs.shape[1] == 0:
            return []

        segments = []
        num_frames = probs.shape[1]
        num_speakers_total = probs.shape[2]
        on_th = onset if onset is not None else threshold
        off_th = offset if offset is not None else on_th

        for spk_idx in range(num_speakers_total):
            spk_probs = probs[0, :, spk_idx]
            is_active = False
            start_frame = 0
            spk_turns = []

            for f in range(num_frames):
                p = spk_probs[f]
                if not is_active:
                    if p >= on_th:
                        is_active = True
                        start_frame = f
                else:
                    if p < off_th:
                        is_active = False
                        end_frame = f
                        start_t = round(start_frame * NEMOTRON_FRAME_DURATION, 3)
                        end_t = round(end_frame * NEMOTRON_FRAME_DURATION, 3)
                        if end_t > start_t:
                            spk_turns.append((start_t, end_t))

            if is_active:
                start_t = round(start_frame * NEMOTRON_FRAME_DURATION, 3)
                end_t = round(num_frames * NEMOTRON_FRAME_DURATION, 3)
                if end_t > start_t:
                    spk_turns.append((start_t, end_t))

            # Filter out turns shorter than min_duration_on
            if min_duration_on > 0:
                spk_turns = [seg for seg in spk_turns if (seg[1] - seg[0]) >= min_duration_on]

            if not spk_turns:
                continue

            # Merge gaps smaller than min_duration_off
            if min_duration_off > 0 and len(spk_turns) > 1:
                merged = [spk_turns[0]]
                for curr_s, curr_e in spk_turns[1:]:
                    prev_s, prev_e = merged[-1]
                    if (curr_s - prev_e) < min_duration_off:
                        merged[-1] = (prev_s, max(prev_e, curr_e))
                    else:
                        merged.append((curr_s, curr_e))
                spk_turns = merged

            for s, e in spk_turns:
                segments.append({
                    "start": round(s, 3),
                    "end": round(e, 3),
                    "speaker": f"SPEAKER_{spk_idx:02d}",
                })

        segments.sort(key=lambda x: (x["start"], x["end"]))

        # Apply speaker filtering if requested
        if num_speakers is not None or max_speakers is not None:
            target_speakers = num_speakers if num_speakers is not None else max_speakers
            durations: Dict[str, float] = {}
            for seg in segments:
                spk = str(seg["speaker"])
                durations[spk] = durations.get(spk, 0.0) + (float(seg["end"]) - float(seg["start"]))

            sorted_spks = sorted(durations.keys(), key=lambda k: durations[k], reverse=True)
            keep_spks = set(sorted_spks[:target_speakers])
            segments = [seg for seg in segments if seg["speaker"] in keep_spks]

        return segments

    def to_rttm(self, segments: List[Dict[str, Union[float, str]]], uri: str = "audio") -> str:
        """Convert speaker turns to NIST RTTM format."""
        return segments_to_rttm(segments, uri=uri)


# Friendly alias
Nemotron3Diarization = NemotronDiarization


def extract_speaker_dict(probs: np.ndarray, threshold: float = 0.5) -> List[Dict[str, Union[float, int]]]:
    """
    Extract speaker turns dictionary from frame-level probabilities.

    :param np.ndarray probs: Array of shape (1, num_frames, num_speakers)
    :param float threshold: Activation threshold (default: 0.5)
    :return: List of dicts with 'Start', 'End', and 'Speaker' (int) keys.
    """
    active = (probs > threshold).astype(np.int32)
    boundary = np.zeros((active.shape[0], 1, active.shape[2]), dtype=np.int32)
    changes = np.diff(np.concatenate([boundary, active, boundary], axis=1), axis=1)

    segments = []
    for speaker in range(changes.shape[2]):
        starts = np.nonzero(changes[0, :, speaker] == 1)[0]
        ends = np.nonzero(changes[0, :, speaker] == -1)[0]
        for start, end in zip(starts, ends):
            segments.append({
                "Start": round(start * NEMOTRON_FRAME_DURATION, 2),
                "End": round(end * NEMOTRON_FRAME_DURATION, 2),
                "Speaker": int(speaker),
            })
    segments.sort(key=lambda seg: (seg["Start"], seg["Speaker"]))
    return segments


def segments_to_rttm(
    segments: List[Dict[str, Union[float, str]]],
    uri: str = "audio",
) -> str:
    """
    Convert diarization segments to standard NIST RTTM string format.

    :param list segments: List of segment dicts (with 'start'/'end'/'speaker' or 'Start'/'End'/'Speaker')
    :param str uri: Recording URI / identifier
    :return: Formatted RTTM string
    """
    lines = []
    for seg in segments:
        s = float(seg.get("start", seg.get("Start", 0.0)))
        e = float(seg.get("end", seg.get("End", 0.0)))
        dur = round(e - s, 3)
        if dur <= 0:
            continue
        spk = seg.get("speaker", seg.get("Speaker", "speaker_00"))
        if isinstance(spk, int):
            spk_label = f"speaker_{spk:02d}"
        else:
            spk_label = str(spk).lower()
        lines.append(f"SPEAKER {uri} 1 {s:.3f} {dur:.3f} <NA> <NA> {spk_label} <NA> <NA>")
    return "\n".join(lines) + ("\n" if lines else "")


class Diarization:
    """
    ONNX-based Speaker Diarization supporting:
    - NVIDIA Nemotron-3 Diarization ("nemotron-3-diarization" / "joosthel/Nemotron-3-Diarization-ONNX") (default)
    - Pyannote Segmentation 3.0 ("pyannote_segmentation")
    """

    def __init__(
        self,
        model: str = "nemotron-3-diarization",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        precision: str = "int8",
        threads: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        :param str model: Diarization model identifier:
            * "nemotron-3-diarization" (default) / "nemotron_diarization" / "joosthel/Nemotron-3-Diarization-ONNX"
            * "pyannote_segmentation" - Pyannote Segmentation 3.0
        :param Optional[str] model_path: Explicit path to ONNX model file or directory
        :param Optional[str] device: Inference device ("cpu", "cuda", "auto")
        :param str precision: Model precision for Nemotron ('int8' or 'fp32', default: 'int8')
        :param Optional[int] threads: Intra-op num threads for onnxruntime
        """
        if ort is None:
            raise ImportError(
                "onnxruntime is required for ONNX diarization. "
                "Install it with: pip install onnxruntime"
            )

        self.model_name = model
        self.device = device or "auto"
        self.precision = precision

        if self.model_name.lower() in [m.lower() for m in NEMOTRON_DIARIZATION_MODELS]:
            self.is_nemotron = True
            self.engine = NemotronDiarization(
                model_dir=model_path,
                precision=precision,
                device=self.device,
                threads=threads,
                **kwargs,
            )
        else:
            self.is_nemotron = False
            if model_path is not None and os.path.exists(model_path):
                self.model_path = model_path
            else:
                self.model_path = get_diarization_model_files()

            self.session = self._init_session(self.model_path, self.device)
            self.input_name = self.session.get_inputs()[0].name

    def _init_session(self, path: str, device: str) -> ort.InferenceSession:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3

        avail = ort.get_available_providers()
        providers = []
        if device in ("cuda", "gpu"):
            if "CUDAExecutionProvider" in avail:
                providers.append("CUDAExecutionProvider")
            else:
                providers.append("CPUExecutionProvider")
        elif device == "cpu":
            providers.append("CPUExecutionProvider")
        else:  # auto
            if "CUDAExecutionProvider" in avail:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

        return ort.InferenceSession(path, sess_options=sess_options, providers=providers)

    def _infer_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Run inference on a 10s audio chunk (160,000 samples).
        Returns marginal speaker probabilities of shape (num_frames, 3).
        """
        if len(chunk) < _WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, _WINDOW_SAMPLES - len(chunk)), mode="constant")
        elif len(chunk) > _WINDOW_SAMPLES:
            chunk = chunk[:_WINDOW_SAMPLES]

        # Shape: (1, 1, 160000)
        tensor = chunk[None, None, :].astype(np.float32)
        outputs = self.session.run(None, {self.input_name: tensor})
        logits = outputs[0][0]  # (num_frames, 7)

        return _powerset_to_multilabel(logits)

    def _sliding_window_inference(
        self,
        audio: np.ndarray,
        window_sec: float = _WINDOW_SECONDS,
        step_sec: float = 5.0,
    ) -> np.ndarray:
        """
        Run sliding window inference with 50% overlap and permutation matching.
        """
        total_samples = len(audio)
        step_samples = int(_SAMPLE_RATE * step_sec)

        if total_samples <= _WINDOW_SAMPLES:
            chunk_probs = self._infer_chunk(audio)
            valid_frames = max(1, min(len(chunk_probs), int((total_samples - _OFFSET_SAMPLES) / _STEP_SAMPLES) + 1))
            return chunk_probs[:valid_frames]

        accumulated_probs = self._infer_chunk(audio[:_WINDOW_SAMPLES])
        overlap_samples = _WINDOW_SAMPLES - step_samples
        overlap_frames = int(overlap_samples / _STEP_SAMPLES)

        pos = step_samples
        while pos < total_samples:
            chunk = audio[pos : pos + _WINDOW_SAMPLES]
            curr_probs = self._infer_chunk(chunk)

            overlap_m = min(overlap_frames, len(accumulated_probs), len(curr_probs))
            if overlap_m > 0:
                prev_overlap = accumulated_probs[-overlap_m:, :3]
                curr_overlap = curr_probs[:overlap_m, :3]

                best_diff = float("inf")
                best_perm = (0, 1, 2)
                for perm in permutations(range(3)):
                    diff = np.mean(np.abs(curr_overlap[:, perm] - prev_overlap))
                    if diff < best_diff:
                        best_diff = diff
                        best_perm = perm

                curr_probs = curr_probs[:, best_perm]

                alpha = np.linspace(0.0, 1.0, overlap_m)[:, None]
                accumulated_probs[-overlap_m:, :3] = (
                    (1.0 - alpha) * accumulated_probs[-overlap_m:, :3] + alpha * curr_probs[:overlap_m, :3]
                )

                new_frames = curr_probs[overlap_m:]
                if len(new_frames) > 0:
                    accumulated_probs = np.vstack([accumulated_probs, new_frames])
            else:
                accumulated_probs = np.vstack([accumulated_probs, curr_probs])

            pos += step_samples

        valid_frames = max(1, min(len(accumulated_probs), int((total_samples - _OFFSET_SAMPLES) / _STEP_SAMPLES) + 1))
        return accumulated_probs[:valid_frames]

    def predict_proba(
        self,
        data: Union[str, Path, np.ndarray],
        sampling_rate: int = _SAMPLE_RATE,
    ) -> np.ndarray:
        """
        Get frame-level speaker probabilities.
        """
        if getattr(self, "is_nemotron", False):
            return self.engine.predict_proba(data, sampling_rate=sampling_rate)

        if isinstance(data, (str, Path)):
            audio = load_audio(data, target_sr=_SAMPLE_RATE)
        elif isinstance(data, np.ndarray):
            audio = data.flatten().astype(np.float32)
            if sampling_rate != _SAMPLE_RATE:
                audio = resample_audio(audio, orig_sr=sampling_rate, target_sr=_SAMPLE_RATE)
        else:
            raise TypeError(f"Unsupported data type for diarization: {type(data)}")

        if len(audio) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        return self._sliding_window_inference(audio)

    def diarize(
        self,
        data: Union[str, Path, np.ndarray],
        sampling_rate: int = _SAMPLE_RATE,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        onset: float = 0.5,
        offset: float = 0.5,
        min_duration_on: float = 0.3,
        min_duration_off: float = 0.5,
        **kwargs,
    ) -> List[Dict[str, Union[float, str]]]:
        """
        Diarize audio data and return list of speaker turns.
        """
        if getattr(self, "is_nemotron", False):
            return self.engine.diarize(
                data=data,
                sampling_rate=sampling_rate,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                onset=onset,
                offset=offset,
                min_duration_on=min_duration_on,
                min_duration_off=min_duration_off,
                **kwargs,
            )

        if isinstance(data, (str, Path)):
            audio = load_audio(data, target_sr=_SAMPLE_RATE)
        elif isinstance(data, np.ndarray):
            audio = data.flatten().astype(np.float32)
            if sampling_rate != _SAMPLE_RATE:
                audio = resample_audio(audio, orig_sr=sampling_rate, target_sr=_SAMPLE_RATE)
        else:
            raise TypeError(f"Unsupported data type for diarization: {type(data)}")

        if len(audio) == 0:
            return []

        speaker_probs = self._sliding_window_inference(audio)
        segments = _binarize_timeline(
            speaker_probs=speaker_probs,
            total_samples=len(audio),
            sample_rate=_SAMPLE_RATE,
            onset=onset,
            offset=offset,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )

        if num_speakers is not None or max_speakers is not None:
            target_speakers = num_speakers if num_speakers is not None else max_speakers
            durations: Dict[str, float] = {}
            for seg in segments:
                spk = str(seg["speaker"])
                durations[spk] = durations.get(spk, 0.0) + (float(seg["end"]) - float(seg["start"]))

            sorted_spks = sorted(durations.keys(), key=lambda k: durations[k], reverse=True)
            keep_spks = set(sorted_spks[:target_speakers])
            segments = [seg for seg in segments if seg["speaker"] in keep_spks]

        return segments


# Module-level cache
_diarizer_instance: Optional[Diarization] = None
_diarizer_model: str = "nemotron-3-diarization"
_diarizer_device: Optional[str] = None
_diarizer_precision: str = "int8"



def merge_same_speaker_segments(
    segments: List[Dict[str, Union[float, str]]],
    max_gap: float = 0.5,
) -> List[Dict[str, Union[float, str]]]:
    """
    Merge adjacent segments spoken by the same speaker if the gap between them
    is less than or equal to max_gap seconds.
    """
    if not segments:
        return []

    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        if prev["speaker"] == seg["speaker"] and (float(seg["start"]) - float(prev["end"])) <= max_gap:
            prev["end"] = max(float(prev["end"]), float(seg["end"]))
        else:
            merged.append(dict(seg))

    return merged


def _run_sherpa_onnx_diarize(
    data: Union[str, Path, np.ndarray],
    sampling_rate: int = _SAMPLE_RATE,
    num_speakers: Optional[int] = None,
    **kwargs,
) -> List[Dict[str, Union[float, str]]]:
    """Fallback runner using sherpa-onnx if requested and available."""
    try:
        import sherpa_onnx
    except ImportError:
        raise ImportError(
            "sherpa-onnx is not installed. Install it with: pip install sherpa-onnx "
            "or use the default backend='onnx'."
        )

    if isinstance(data, (str, Path)):
        audio = load_audio(data, target_sr=_SAMPLE_RATE)
    else:
        audio = np.asarray(data, dtype=np.float32).flatten()
        if sampling_rate != _SAMPLE_RATE:
            audio = resample_audio(audio, orig_sr=sampling_rate, target_sr=_SAMPLE_RATE)

    model_dir = kwargs.get("model_dir", None)
    seg_model_path = get_diarization_model_files(model_dir=model_dir)

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=seg_model_path
            )
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers is not None else 0
        ),
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    result = sd.process(audio, sample_rate=_SAMPLE_RATE)

    segments = []
    for seg in result:
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "speaker": f"SPEAKER_{seg.speaker:02d}",
        })
    return segments


def diarize(
    data: Union[str, Path, np.ndarray],
    model: str = "nemotron-3-diarization",
    device: Optional[str] = None,
    precision: str = "int8",
    sampling_rate: int = _SAMPLE_RATE,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    onset: float = 0.5,
    offset: float = 0.5,
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.5,
    backend: str = "onnx",
    **kwargs,
) -> List[Dict[str, Union[float, str]]]:
    """
    Perform speaker diarization on audio data using ONNX.

    :param Union[str, Path, np.ndarray] data: Path to sound file or numpy array of audio.
    :param str model: Diarization model name (default: "nemotron-3-diarization").
        Options: "nemotron-3-diarization", "pyannote_segmentation", etc.
    :param Optional[str] device: Inference device ("cpu", "cuda", "auto").
    :param str precision: Model precision for Nemotron ('int8' or 'fp32', default: 'int8').
    :param int sampling_rate: Audio sample rate (default: 16000).
    :param Optional[int] num_speakers: Exact number of speakers if known.
    :param Optional[int] min_speakers: Minimum number of speakers.
    :param Optional[int] max_speakers: Maximum number of speakers.
    :param float onset: Speech onset probability threshold (default: 0.5).
    :param float offset: Speech offset probability threshold (default: 0.5).
    :param float min_duration_on: Minimum speaker turn duration in seconds (default: 0.3).
    :param float min_duration_off: Minimum silence duration to split turns in seconds (default: 0.5).
    :param str backend: Diarization backend ("onnx" or "sherpa-onnx", default: "onnx").
    :return: List of speaker segments with 'start', 'end', and 'speaker' keys.
    :rtype: List[Dict[str, Union[float, str]]]

    **Example:**
        .. code-block:: python

            from pythaiasr import diarize

            segments = diarize("conversation.wav")
            for seg in segments:
                print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}")
    """
    if backend == "sherpa-onnx":
        return _run_sherpa_onnx_diarize(
            data=data,
            sampling_rate=sampling_rate,
            num_speakers=num_speakers,
            **kwargs,
        )

    if backend != "onnx":
        raise ValueError(f"Unknown backend '{backend}'. Supported backends are 'onnx' and 'sherpa-onnx'.")

    global _diarizer_instance, _diarizer_model, _diarizer_device, _diarizer_precision
    if (
        _diarizer_instance is None
        or _diarizer_model != model
        or _diarizer_device != device
        or _diarizer_precision != precision
    ):
        _diarizer_instance = Diarization(model=model, device=device, precision=precision, **kwargs)
        _diarizer_model = model
        _diarizer_device = device
        _diarizer_precision = precision

    return _diarizer_instance.diarize(
        data=data,
        sampling_rate=sampling_rate,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        onset=onset,
        offset=offset,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )


def asr_diarize(
    data: Union[str, Path, np.ndarray],
    asr_model: str = "typhoon_asr",
    diarize_model: str = "nemotron-3-diarization",
    device: Optional[str] = None,
    precision: str = "int8",
    sampling_rate: int = _SAMPLE_RATE,
    lm: bool = False,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    onset: float = 0.5,
    offset: float = 0.5,
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.5,
    merge_same_speaker: bool = True,
    max_merge_gap: float = 0.5,
    backend: str = "onnx",
    **kwargs,
) -> List[Dict[str, Union[float, str]]]:
    """
    Perform speech diarization and Automatic Speech Recognition (ASR) on audio.
    Transcribes each detected speaker segment and attributes the text to the speaker.

    :param Union[str, Path, np.ndarray] data: Path to sound file or numpy array of audio.
    :param str asr_model: The ASR model name (default: "typhoon_asr").
    :param str diarize_model: Diarization model name (default: "nemotron-3-diarization").
    :param Optional[str] device: Inference device ("cpu", "cuda", "auto").
    :param str precision: Model precision for Nemotron ('int8' or 'fp32', default: 'int8').
    :param int sampling_rate: Audio sample rate (default: 16000).
    :param bool lm: Use language model for ASR if supported.
    :param Optional[int] num_speakers: Exact number of speakers if known.
    :param Optional[int] min_speakers: Minimum number of speakers.
    :param Optional[int] max_speakers: Maximum number of speakers.
    :param float onset: Speech onset probability threshold (default: 0.5).
    :param float offset: Speech offset probability threshold (default: 0.5).
    :param float min_duration_on: Minimum speaker turn duration in seconds (default: 0.3).
    :param float min_duration_off: Minimum silence duration to split turns in seconds (default: 0.5).
    :param bool merge_same_speaker: Whether to merge consecutive segments from the same speaker.
    :param float max_merge_gap: Maximum silence duration in seconds between same-speaker segments to merge.
    :param str backend: Diarization backend ("onnx" or "sherpa-onnx", default: "onnx").
    :return: List of speaker turns with 'start', 'end', 'speaker', and 'text' keys.
    :rtype: List[Dict[str, Union[float, str]]]

    **Example:**
        .. code-block:: python

            from pythaiasr import asr_diarize

            turns = asr_diarize("meeting.wav", asr_model="typhoon_asr", diarize_model="nemotron-3-diarization")
            for turn in turns:
                print(f"[{turn['start']:.2f}s - {turn['end']:.2f}s] {turn['speaker']}: {turn['text']}")
    """
    from pythaiasr import asr

    # Load audio array at target sampling rate
    if isinstance(data, (str, Path)):
        audio = load_audio(data, target_sr=sampling_rate)
    elif isinstance(data, np.ndarray):
        audio = data.flatten().astype(np.float32)
        if sampling_rate != _SAMPLE_RATE:
            audio = resample_audio(audio, orig_sr=sampling_rate, target_sr=_SAMPLE_RATE)
            sampling_rate = _SAMPLE_RATE
    else:
        raise TypeError(f"Unsupported data type for asr_diarize: {type(data)}")

    # 1. Run speaker diarization
    segments = diarize(
        data=audio,
        model=diarize_model,
        device=device,
        precision=precision,
        sampling_rate=sampling_rate,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        onset=onset,
        offset=offset,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
        backend=backend,
        **kwargs,
    )

    if not segments:
        return []

    # 2. Optionally merge consecutive segments of the same speaker
    if merge_same_speaker:
        segments = merge_same_speaker_segments(segments, max_gap=max_merge_gap)

    # 3. Transcribe each speaker turn
    results = []
    total_audio_len = len(audio)

    for seg in segments:
        s_sec = float(seg["start"])
        e_sec = float(seg["end"])
        start_idx = max(0, int(s_sec * sampling_rate))
        end_idx = min(total_audio_len, int(e_sec * sampling_rate))

        chunk = audio[start_idx:end_idx]
        if len(chunk) > 0:
            text = asr(
                data=chunk,
                model=asr_model,
                lm=lm,
                device=device,
                sampling_rate=sampling_rate,
            )
        else:
            text = ""

        results.append({
            "start": s_sec,
            "end": e_sec,
            "speaker": seg["speaker"],
            "text": text.strip() if isinstance(text, str) else str(text),
        })

    return results


__all__ = [
    "Diarization",
    "NemotronDiarization",
    "Nemotron3Diarization",
    "diarize",
    "asr_diarize",
    "merge_same_speaker_segments",
    "segments_to_rttm",
    "extract_speaker_dict",
]


