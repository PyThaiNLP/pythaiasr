# -*- coding: utf-8 -*-
"""Unit tests for ONNX-based speech diarization (diarize and asr_diarize)."""

import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from pythaiasr import (
    Diarization,
    diarize,
    asr_diarize,
    merge_same_speaker_segments,
    get_diarization_model_files,
)
from pythaiasr.diarization import (
    _powerset_to_multilabel,
    _binarize_timeline,
    _SAMPLE_RATE,
    _STEP_SAMPLES,
    _OFFSET_SAMPLES,
)

TEST_WAV_FILE = os.path.join(".", "tests", "test.wav")
COMMON_VOICE_FILE = os.path.join(".", "tests", "common_voice_th_25686161.wav")


class TestDiarizationModule(unittest.TestCase):
    """Test suite for Diarization functions and math."""

    def test_imports(self):
        """Verify that all public diarization symbols are callable and exported."""
        self.assertTrue(callable(Diarization))
        self.assertTrue(callable(diarize))
        self.assertTrue(callable(asr_diarize))
        self.assertTrue(callable(merge_same_speaker_segments))
        self.assertTrue(callable(get_diarization_model_files))

    def test_powerset_to_multilabel_math(self):
        """Test conversion of 7-class powerset logits to 3-speaker marginal probabilities."""
        # Case 1: Pure silence / non-speech (class 0 has large logit)
        logits_silence = np.array([[[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
        probs_silence = _powerset_to_multilabel(logits_silence)[0, 0]
        self.assertEqual(probs_silence.shape, (3,))
        self.assertTrue(np.all(probs_silence < 0.01))

        # Case 2: Speaker 1 only (class 1 has large logit)
        logits_spk1 = np.array([[[0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
        probs_spk1 = _powerset_to_multilabel(logits_spk1)[0, 0]
        self.assertGreater(probs_spk1[0], 0.99)
        self.assertLess(probs_spk1[1], 0.01)
        self.assertLess(probs_spk1[2], 0.01)

        # Case 3: Overlapping Speaker 1 & Speaker 2 (class 4 has large logit)
        logits_spk12 = np.array([[[0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0]]], dtype=np.float32)
        probs_spk12 = _powerset_to_multilabel(logits_spk12)[0, 0]
        self.assertGreater(probs_spk12[0], 0.99)
        self.assertGreater(probs_spk12[1], 0.99)
        self.assertLess(probs_spk12[2], 0.01)

        # Case 4: Overlapping Speaker 2 & Speaker 3 (class 6 has large logit)
        logits_spk23 = np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]]], dtype=np.float32)
        probs_spk23 = _powerset_to_multilabel(logits_spk23)[0, 0]
        self.assertLess(probs_spk23[0], 0.01)
        self.assertGreater(probs_spk23[1], 0.99)
        self.assertGreater(probs_spk23[2], 0.99)

    def test_binarize_timeline(self):
        """Test hysteresis thresholding, min duration, and gap merging in timeline binarization."""
        # 300 frames (~5.0 seconds at step 270)
        num_frames = 300
        total_samples = int(num_frames * _STEP_SAMPLES + _OFFSET_SAMPLES)
        probs = np.zeros((num_frames, 3), dtype=np.float32)

        # Speaker 0 active from frame 20 to 60 (~0.34s to ~1.01s)
        probs[20:60, 0] = 0.9
        # Brief spike for speaker 0 from frame 70 to 72 (~34ms, should be filtered by min_duration_on=0.3s)
        probs[70:73, 0] = 0.9
        # Speaker 1 active from frame 100 to 140, brief pause 141-145, then 146 to 180 (should be merged)
        probs[100:140, 1] = 0.9
        probs[145:180, 1] = 0.9

        segments = _binarize_timeline(
            speaker_probs=probs,
            total_samples=total_samples,
            sample_rate=_SAMPLE_RATE,
            onset=0.5,
            offset=0.5,
            min_duration_on=0.3,
            min_duration_off=0.5,
        )

        spk0_segs = [s for s in segments if s["speaker"] == "SPEAKER_00"]
        spk1_segs = [s for s in segments if s["speaker"] == "SPEAKER_01"]

        # Speaker 0 should have 1 segment (the spike was dropped)
        self.assertEqual(len(spk0_segs), 1)
        self.assertAlmostEqual(spk0_segs[0]["start"], (20 * _STEP_SAMPLES) / _SAMPLE_RATE, places=2)

        # Speaker 1 should have 1 merged segment
        self.assertEqual(len(spk1_segs), 1)
        self.assertAlmostEqual(spk1_segs[0]["start"], (100 * _STEP_SAMPLES) / _SAMPLE_RATE, places=2)

    def test_merge_same_speaker_segments(self):
        """Test merge_same_speaker_segments helper function."""
        raw_segments = [
            {"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"},
            {"start": 1.4, "end": 2.5, "speaker": "SPEAKER_00"},  # Gap 0.2s <= 0.5s -> should merge
            {"start": 2.7, "end": 4.0, "speaker": "SPEAKER_01"},  # Different speaker -> no merge
            {"start": 4.8, "end": 6.0, "speaker": "SPEAKER_01"},  # Gap 0.8s > 0.5s -> no merge
        ]

        merged = merge_same_speaker_segments(raw_segments, max_gap=0.5)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["speaker"], "SPEAKER_00")
        self.assertEqual(merged[0]["start"], 0.0)
        self.assertEqual(merged[0]["end"], 2.5)

        self.assertEqual(merged[1]["speaker"], "SPEAKER_01")
        self.assertEqual(merged[1]["start"], 2.7)
        self.assertEqual(merged[1]["end"], 4.0)

        self.assertEqual(merged[2]["speaker"], "SPEAKER_01")
        self.assertEqual(merged[2]["start"], 4.8)
        self.assertEqual(merged[2]["end"], 6.0)

    def test_mock_diarization_short_and_long_audio(self):
        """Test Diarization class sliding window and inference with a mocked ONNX session."""
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]

        # Return dummy 7-class logits (shape: 1, 589, 7)
        def mock_run(output_names, input_feed):
            batch_size = 1
            num_frames = 589
            logits = np.zeros((batch_size, num_frames, 7), dtype=np.float32)
            # Make speaker 1 active in middle frames
            logits[:, 100:200, 1] = 5.0
            return [logits]

        mock_session.run.side_effect = mock_run

        with patch.object(Diarization, "_init_session", return_value=mock_session):
            with patch("pythaiasr.diarization.get_diarization_model_files", return_value="dummy_path.onnx"):
                diarizer = Diarization()

                # 1. Short audio (3 seconds = 48,000 samples)
                short_audio = np.random.randn(48000).astype(np.float32)
                short_segs = diarizer.diarize(short_audio)
                self.assertIsInstance(short_segs, list)
                for seg in short_segs:
                    self.assertIn("start", seg)
                    self.assertIn("end", seg)
                    self.assertIn("speaker", seg)
                    self.assertLessEqual(seg["end"], 3.05)

                # 2. Long audio (15 seconds = 240,000 samples, requires sliding window)
                long_audio = np.random.randn(240000).astype(np.float32)
                long_segs = diarizer.diarize(long_audio)
                self.assertIsInstance(long_segs, list)

                # 3. Test num_speakers filtering
                filtered_segs = diarizer.diarize(long_audio, num_speakers=1)
                speakers = {s["speaker"] for s in filtered_segs}
                self.assertLessEqual(len(speakers), 1)

    def test_diarize_backend_validation(self):
        """Test backend validation and error handling."""
        audio = np.zeros(16000, dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            diarize(audio, backend="unsupported_backend")
        self.assertIn("Unknown backend", str(ctx.exception))

    def test_sherpa_onnx_backend_import_error(self):
        """Test that requesting sherpa-onnx backend raises informative error when missing."""
        import sys
        sherpa_backup = sys.modules.get("sherpa_onnx")
        try:
            sys.modules["sherpa_onnx"] = None
            with self.assertRaises(ImportError) as ctx:
                diarize(np.zeros(16000, dtype=np.float32), backend="sherpa-onnx")
            self.assertIn("sherpa-onnx is not installed", str(ctx.exception))
        finally:
            if sherpa_backup is not None:
                sys.modules["sherpa_onnx"] = sherpa_backup
            else:
                sys.modules.pop("sherpa_onnx", None)

    @patch("pythaiasr.diarization.diarize")
    @patch("pythaiasr.asr")
    def test_asr_diarize_integration(self, mock_asr, mock_diarize):
        """Test asr_diarize pipeline integrating diarize and asr."""
        # Mock diarize return
        mock_diarize.return_value = [
            {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
            {"start": 1.6, "end": 3.0, "speaker": "SPEAKER_01"},
        ]

        # Mock asr return
        mock_asr.side_effect = ["สวัสดีครับ", "ยินดีต้อนรับครับ"]

        audio = np.random.randn(48000).astype(np.float32)
        turns = asr_diarize(
            data=audio,
            asr_model="typhoon_asr",
            merge_same_speaker=False,
        )

        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["speaker"], "SPEAKER_00")
        self.assertEqual(turns[0]["text"], "สวัสดีครับ")
        self.assertEqual(turns[0]["start"], 0.0)
        self.assertEqual(turns[0]["end"], 1.5)

        self.assertEqual(turns[1]["speaker"], "SPEAKER_01")
        self.assertEqual(turns[1]["text"], "ยินดีต้อนรับครับ")
        self.assertEqual(turns[1]["start"], 1.6)
        self.assertEqual(turns[1]["end"], 3.0)

    def test_empty_audio_handling(self):
        """Test diarize and asr_diarize with empty audio array."""
        empty_audio = np.array([], dtype=np.float32)
        mock_session = MagicMock()
        with patch.object(Diarization, "_init_session", return_value=mock_session):
            with patch("pythaiasr.diarization.get_diarization_model_files", return_value="dummy.onnx"):
                diarizer = Diarization()
                res = diarizer.diarize(empty_audio)
                self.assertEqual(res, [])

                res_asr = asr_diarize(empty_audio)
                self.assertEqual(res_asr, [])

    @patch("pythaiasr.diarization.diarize")
    @patch("pythaiasr.asr")
    def test_asr_diarize_merge_same_speaker(self, mock_asr, mock_diarize):
        """Test asr_diarize merging consecutive turns of the same speaker."""
        mock_diarize.return_value = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.2, "end": 2.0, "speaker": "SPEAKER_00"},  # Same speaker, gap 0.2s <= 0.5s -> merged!
            {"start": 2.5, "end": 3.5, "speaker": "SPEAKER_01"},  # Different speaker
        ]
        mock_asr.side_effect = ["สวัสดีครับยินดีต้อนรับ", "ขอบคุณครับ"]

        audio = np.zeros(16000 * 4, dtype=np.float32)
        turns = asr_diarize(audio, merge_same_speaker=True, max_merge_gap=0.5)

        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["speaker"], "SPEAKER_00")
        self.assertEqual(turns[0]["start"], 0.0)
        self.assertEqual(turns[0]["end"], 2.0)
        self.assertEqual(turns[0]["text"], "สวัสดีครับยินดีต้อนรับ")

        self.assertEqual(turns[1]["speaker"], "SPEAKER_01")
        self.assertEqual(turns[1]["start"], 2.5)
        self.assertEqual(turns[1]["end"], 3.5)
        self.assertEqual(turns[1]["text"], "ขอบคุณครับ")

    def test_diarize_sampling_rate_conversion(self):
        """Test that non-16kHz numpy arrays are automatically resampled."""
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]

        # 24kHz audio (2 seconds = 48,000 samples)
        audio_24k = np.random.randn(48000).astype(np.float32)

        def mock_run(output_names, input_feed):
            inp = input_feed["input"]
            # After resampling to 16kHz, length should be 32,000 samples padded to 160,000
            self.assertEqual(inp.shape[-1], 160000)
            logits = np.zeros((1, 589, 7), dtype=np.float32)
            logits[:, 20:80, 1] = 5.0
            return [logits]

        mock_session.run.side_effect = mock_run

        with patch.object(Diarization, "_init_session", return_value=mock_session):
            with patch("pythaiasr.diarization.get_diarization_model_files", return_value="dummy.onnx"):
                diarizer = Diarization()
                segments = diarizer.diarize(audio_24k, sampling_rate=24000)
                self.assertIsInstance(segments, list)
                self.assertGreater(len(segments), 0)

    def test_get_diarization_model_files_resolution(self):
        """Test custom model directory and candidate path resolution for diarization model."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_seg = os.path.join(tmpdir, "segmentation-3.0.onnx")
            with open(dummy_seg, "wb") as f:
                f.write(b"onnx_model_content")

            path = get_diarization_model_files(model_dir=tmpdir)
            self.assertEqual(path, dummy_seg)


if __name__ == "__main__":
    unittest.main()
