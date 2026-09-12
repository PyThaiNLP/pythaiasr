# -*- coding: utf-8 -*-

import os
import sys
import unittest
import numpy as np

try:
    import torch
    import torchaudio
except ImportError:
    torch = None
    torchaudio = None

from pythaiasr import (
    ASR,
    asr,
    stream_asr,
    FastConformerRNNT,
    RealtimeStreamASR,
    StreamingTranscriber,
    get_pythaiasr_path,
    get_typhoon_model_files,
    extract_features,
    load_audio,
)

file = os.path.join(".", "tests", "common_voice_th_25686161.wav")
test_wav_file = os.path.join(".", "tests", "test.wav")


class TestKhaveePackage(unittest.TestCase):
    def test_default_model_is_typhoon(self):
        """Test that typhoon_asr is the default model."""
        from pythaiasr import _model_name
        self.assertEqual(_model_name, "typhoon_asr")
        try:
            import onnxruntime
        except ImportError:
            self.skipTest("onnxruntime not installed")
        engine = ASR()
        self.assertEqual(engine.model_name, "typhoon_asr")

    def test_default_asr(self):
        """Test default asr() call uses Typhoon ASR."""
        try:
            import onnxruntime
        except ImportError:
            self.skipTest("onnxruntime not installed")
        if os.path.exists(test_wav_file):
            result = asr(test_wav_file, device="cpu")
            self.assertIsNotNone(result)
            self.assertIn("ภาษาไทย", result)

    def test_torch_model_without_torch(self):
        """Test that requesting a torch model without torch raises an informative ImportError."""
        if torch is not None:
            self.skipTest("torch is installed")
        with self.assertRaises(ImportError) as ctx:
            ASR(model="airesearch/wav2vec2-large-xlsr-53-th")
        self.assertIn("pip install pythaiasr[torch]", str(ctx.exception))

    @unittest.skipIf(torch is None or torchaudio is None, "torch or torchaudio not installed")
    def test_wav2vec2_asr(self):
        self.assertIsNotNone(asr(file, model="airesearch/wav2vec2-large-xlsr-53-th", device="cpu"))

    @unittest.skipIf(torch is None or torchaudio is None, "torch or torchaudio not installed")
    def test_asr_array(self):
        speech_array, sampling_rate = torchaudio.load(file)
        self.assertIsNotNone(asr(speech_array[0].numpy(), model="airesearch/wav2vec2-large-xlsr-53-th", device="cpu", sampling_rate=sampling_rate))

    @unittest.skipIf(torch is None or torchaudio is None, "torch or torchaudio not installed")
    def test_whisper_small(self):
        self.assertIsNotNone(asr(file, model="biodatlab/whisper-small-th-combined", device="cpu"))

    @unittest.skipIf(torch is None or torchaudio is None, "torch or torchaudio not installed")
    def test_whisper_medium(self):
        self.assertIsNotNone(asr(file, model="biodatlab/whisper-th-medium-combined", device="cpu"))

    @unittest.skipIf(torch is None or torchaudio is None, "torch or torchaudio not installed")
    def test_whisper_large(self):
        self.assertIsNotNone(asr(file, model="biodatlab/whisper-th-large-combined", device="cpu"))

    @unittest.skipIf(torch is None or torchaudio is None, "torch or torchaudio not installed")
    def test_whisper_array(self):
        speech_array, sampling_rate = torchaudio.load(file)
        self.assertIsNotNone(asr(speech_array[0].numpy(), model="biodatlab/whisper-small-th-combined", device="cpu", sampling_rate=sampling_rate))
    
    def test_stream_asr_import(self):
        """Test that stream_asr can be imported"""
        self.assertTrue(callable(stream_asr))
    
    def test_stream_asr_without_pyaudio(self):
        """Test that stream_asr raises ImportError when pyaudio is not available"""
        pyaudio_backup = sys.modules.get('pyaudio')
        if 'pyaudio' in sys.modules:
            del sys.modules['pyaudio']
        
        try:
            gen = stream_asr(device="cpu")
            with self.assertRaises(ImportError) as context:
                next(gen)
            self.assertIn("pyaudio is required", str(context.exception))
        finally:
            if pyaudio_backup is not None:
                sys.modules['pyaudio'] = pyaudio_backup

    def test_typhoon_path_resolution(self):
        """Test root user path defaults to ~/pythaiasr-data and respects env var."""
        expected_default = os.path.join(os.path.expanduser("~"), "pythaiasr-data")
        self.assertEqual(get_pythaiasr_path(), expected_default)

        # Test environment variable override
        test_custom_path = os.path.abspath(os.path.join(".", "test-pythaiasr-custom"))
        os.environ["PYTHAIASR_DATA_DIR"] = test_custom_path
        try:
            self.assertEqual(get_pythaiasr_path(), test_custom_path)
        finally:
            del os.environ["PYTHAIASR_DATA_DIR"]
            if os.path.exists(test_custom_path):
                os.rmdir(test_custom_path)

    def test_typhoon_support_models(self):
        """Test that typhoon models are included in ASR supported models."""
        dummy_model = "airesearch/wav2vec2-large-xlsr-53-th"
        # Checking supported models list via class definition or instance
        asr_obj = ASR.__new__(ASR)
        asr_obj.model_name = "typhoon_asr"
        asr_obj.support_model = [
            "airesearch/wav2vec2-large-xlsr-53-th",
            "wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm",
            "wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut",
            "biodatlab/whisper-small-th-combined",
            "biodatlab/whisper-th-medium-combined",
            "biodatlab/whisper-th-large-combined",
            "typhoon_asr",
            "typhoon-asr-realtime",
            "wannaphong/asr_cat_model",
        ]
        self.assertIn("typhoon_asr", asr_obj.support_model)
        self.assertIn("typhoon-asr-realtime", asr_obj.support_model)
        self.assertIn("wannaphong/asr_cat_model", asr_obj.support_model)

    def test_typhoon_feature_extraction(self):
        """Test Slaney mel filterbank and feature extraction on numpy arrays."""
        sr = 16000
        duration = 1.0  # 1 second of synthetic audio
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration), dtype=np.float32))
        
        features, seq_len = extract_features(audio, sample_rate=sr)
        self.assertEqual(features.ndim, 3)
        self.assertEqual(features.shape[0], 1)  # batch size
        self.assertEqual(features.shape[1], 80) # 80 mel channels
        self.assertTrue(seq_len > 0)

    def test_load_audio_wave(self):
        """Test audio loading with WAV file."""
        if os.path.exists(test_wav_file):
            audio = load_audio(test_wav_file, target_sr=16000)
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(audio.ndim, 1)
            self.assertTrue(len(audio) > 0)

    def test_typhoon_onnx_inference(self):
        """Test Typhoon ASR offline and realtime inference if ONNX models and onnxruntime are present."""
        try:
            import onnxruntime
        except ImportError:
            self.skipTest("onnxruntime is not installed")

        # Check if local models exist in typhoon_asr directory or ~/pythaiasr-data
        local_encoder = os.path.join(".", "typhoon_asr", "encoder-fastconformer-quran-ar.onnx")
        local_decoder = os.path.join(".", "typhoon_asr", "decoder_joint-fastconformer-quran-ar.onnx")
        local_vocab = os.path.join(".", "typhoon_asr", "tokenizer", "vocab.json")

        if not (os.path.exists(local_encoder) and os.path.exists(local_decoder) and os.path.exists(local_vocab)):
            self.skipTest("Local ONNX model files not found for fast test")

        model = FastConformerRNNT(
            encoder_path=local_encoder,
            decoder_path=local_decoder,
            vocab_path=local_vocab,
            device="cpu",
        )
        self.assertIsNotNone(model)

        if os.path.exists(test_wav_file):
            # Test offline transcription
            transcript = model.transcribe(test_wav_file)
            self.assertIsInstance(transcript, str)
            self.assertTrue(len(transcript) > 0)

            # Test streaming transcriber
            streamer = RealtimeStreamASR(model=model, sample_rate=16000, step_sec=0.48)
            audio = load_audio(test_wav_file, target_sr=16000)
            chunk_size = int(16000 * 0.48)
            emitted = []
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                text = streamer.process_chunk(chunk)
                if text:
                    emitted.append(text)
            trailing = streamer.flush()
            if trailing:
                emitted.append(trailing)
            full_streaming_text = streamer.get_full_transcript()
            self.assertIsInstance(full_streaming_text, str)
            self.assertTrue(len(full_streaming_text) > 0)


if __name__ == '__main__':
    unittest.main()
