# -*- coding: utf-8 -*-
"""Speech Diarization with ONNX for PyThaiASR.

Provides:
- Diarization: ONNX-based speaker diarization engine using Pyannote Segmentation 3.0.
- diarize: Public function to extract timestamped speaker segments.
- asr_diarize: Public function to extract speaker segments and transcribe them with ASR.
"""

from __future__ import annotations

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

from pythaiasr.download import get_diarization_model_files
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


class Diarization:
    """
    ONNX-based Speaker Diarization using Pyannote Segmentation.
    """

    def __init__(
        self,
        model: str = "pyannote_segmentation",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        :param str model: Diarization model identifier (default: "pyannote_segmentation")
        :param Optional[str] model_path: Explicit path to ONNX model file
        :param Optional[str] device: Inference device ("cpu", "cuda", "auto")
        """
        if ort is None:
            raise ImportError(
                "onnxruntime is required for ONNX diarization. "
                "Install it with: pip install onnxruntime"
            )

        self.model_name = model
        self.device = device or "auto"

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
            # Calculate number of valid frames corresponding to actual audio
            valid_frames = max(1, min(len(chunk_probs), int((total_samples - _OFFSET_SAMPLES) / _STEP_SAMPLES) + 1))
            return chunk_probs[:valid_frames]

        # First chunk
        accumulated_probs = self._infer_chunk(audio[:_WINDOW_SAMPLES])
        overlap_samples = _WINDOW_SAMPLES - step_samples
        # Overlap in output frames
        overlap_frames = int(overlap_samples / _STEP_SAMPLES)

        pos = step_samples
        while pos < total_samples:
            chunk = audio[pos : pos + _WINDOW_SAMPLES]
            curr_probs = self._infer_chunk(chunk)

            # Match local speakers of curr_probs with accumulated_probs on overlap region
            overlap_m = min(overlap_frames, len(accumulated_probs), len(curr_probs))
            if overlap_m > 0:
                prev_overlap = accumulated_probs[-overlap_m:, :3]
                curr_overlap = curr_probs[:overlap_m, :3]

                # Find permutation that minimizes mean absolute difference
                best_diff = float("inf")
                best_perm = (0, 1, 2)
                for perm in permutations(range(3)):
                    diff = np.mean(np.abs(curr_overlap[:, perm] - prev_overlap))
                    if diff < best_diff:
                        best_diff = diff
                        best_perm = perm

                # Apply best permutation to current chunk
                curr_probs = curr_probs[:, best_perm]

                # Blend overlap frames linearly
                alpha = np.linspace(0.0, 1.0, overlap_m)[:, None]
                accumulated_probs[-overlap_m:, :3] = (
                    (1.0 - alpha) * accumulated_probs[-overlap_m:, :3] + alpha * curr_probs[:overlap_m, :3]
                )

                # Append new non-overlapping frames
                new_frames = curr_probs[overlap_m:]
                if len(new_frames) > 0:
                    accumulated_probs = np.vstack([accumulated_probs, new_frames])
            else:
                accumulated_probs = np.vstack([accumulated_probs, curr_probs])

            pos += step_samples

        # Trim to valid frames for total audio length
        valid_frames = max(1, min(len(accumulated_probs), int((total_samples - _OFFSET_SAMPLES) / _STEP_SAMPLES) + 1))
        return accumulated_probs[:valid_frames]

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
    ) -> List[Dict[str, Union[float, str]]]:
        """
        Diarize audio data and return list of speaker turns.
        """
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

        # Apply speaker filtering if requested
        if num_speakers is not None or max_speakers is not None:
            target_speakers = num_speakers if num_speakers is not None else max_speakers
            # Rank speakers by total speech duration
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
_diarizer_model: str = "pyannote_segmentation"
_diarizer_device: Optional[str] = None


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
    model: str = "pyannote_segmentation",
    device: Optional[str] = None,
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
    :param str model: Diarization model name (default: "pyannote_segmentation").
    :param Optional[str] device: Inference device ("cpu", "cuda", "auto").
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

    global _diarizer_instance, _diarizer_model, _diarizer_device
    if _diarizer_instance is None or _diarizer_model != model or _diarizer_device != device:
        _diarizer_instance = Diarization(model=model, device=device)
        _diarizer_model = model
        _diarizer_device = device

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
    diarize_model: str = "pyannote_segmentation",
    device: Optional[str] = None,
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
    :param str diarize_model: Diarization model name (default: "pyannote_segmentation").
    :param Optional[str] device: Inference device ("cpu", "cuda", "auto").
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

            turns = asr_diarize("meeting.wav", asr_model="typhoon_asr")
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

