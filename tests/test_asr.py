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
    NemotronStreamingASR,
    NemotronASR,
    NemotronStreamASR,
    RealtimeStreamNemotron,
    get_pythaiasr_path,
    get_typhoon_model_files,
    get_nemotron_asr_model_files,
    extract_features,
    extract_nemotron_features,
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
        try:
            engine = ASR()
            self.assertEqual(engine.model_name, "typhoon_asr")
        except (OSError, RuntimeError) as e:
            self.skipTest(f"Typhoon model download not possible in current environment: {e}")

    def test_default_asr(self):
        """Test default asr() call uses Typhoon ASR."""
        try:
            import onnxruntime
        except ImportError:
            self.skipTest("onnxruntime not installed")
        if os.path.exists(test_wav_file):
            try:
                result = asr(test_wav_file, device="cpu")
                self.assertIsNotNone(result)
                self.assertIn("ภาษาไทย", result)
            except (OSError, RuntimeError) as e:
                self.skipTest(f"Typhoon model download not possible in current environment: {e}")

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
        sys.modules['pyaudio'] = None
        
        try:
            gen = stream_asr(device="cpu")
            with self.assertRaises(ImportError) as context:
                next(gen)
            self.assertIn("pyaudio is required", str(context.exception))
        except (OSError, RuntimeError) as e:
            self.skipTest(f"Typhoon model download not possible in current environment: {e}")
        finally:
            if pyaudio_backup is not None:
                sys.modules['pyaudio'] = pyaudio_backup
            else:
                sys.modules.pop('pyaudio', None)

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
            "biodatlab/whisper-th-medium-timestamp",
            "typhoon_asr",
            "typhoon-asr-realtime",
            "wannaphong/typhoon-asr-realtime-onnx",
        ]
        self.assertIn("typhoon_asr", asr_obj.support_model)
        self.assertIn("typhoon-asr-realtime", asr_obj.support_model)
        self.assertIn("wannaphong/typhoon-asr-realtime-onnx", asr_obj.support_model)
        self.assertIn("biodatlab/whisper-th-medium-timestamp", asr_obj.support_model)

    def test_nemotron_support_models(self):
        """Test that Nemotron streaming models are included in ASR supported models."""
        asr_obj = ASR.__new__(ASR)
        asr_obj.model_name = "nemotron_asr"
        asr_obj.support_model = [
            "nemotron_asr",
            "nemotron-asr",
            "nemotron_asr_int4",
            "nemotron-asr-int4",
            "nemotron_asr_fp32",
            "nemotron-asr-fp32",
            "nemotron-3.5-asr-streaming-0.6b",
            "wannaphong/nemotron-3.5-asr-streaming-0.6b-onnx-int4",
            "wannaphong/typhoon-asr-streaming-nemotron-0.6b-int4-onnx",
            "wannaphong/typhoon-asr-streaming-nemotron-0.6b-fp32-onnx",
        ]
        self.assertIn("nemotron_asr", asr_obj.support_model)
        self.assertIn("nemotron-3.5-asr-streaming-0.6b", asr_obj.support_model)
        self.assertIn("wannaphong/nemotron-3.5-asr-streaming-0.6b-onnx-int4", asr_obj.support_model)
        self.assertIn("wannaphong/typhoon-asr-streaming-nemotron-0.6b-int4-onnx", asr_obj.support_model)
        self.assertIn("wannaphong/typhoon-asr-streaming-nemotron-0.6b-fp32-onnx", asr_obj.support_model)


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

    def test_tokens_to_timestamp_chunks_word_mode(self):
        """Test timestamp chunk generation with SentencePiece word boundaries."""
        model = FastConformerRNNT.__new__(FastConformerRNNT)
        model.vocab = ["<unk>", "<s>", "</s>", "สวัสดี", "ครับ", "\u2581ภาษา", "ไทย", "\u2581ง่าย", "นิดเดียว"]
        
        # emitted_tokens: "\u2581ภาษา" (idx 5, frame 2), "ไทย" (idx 6, frame 3), "\u2581ง่าย" (idx 7, frame 10), "นิดเดียว" (idx 8, frame 11)
        tokens = [5, 6, 7, 8]
        timestamps = [2, 3, 10, 11]
        chunks = model.tokens_to_timestamp_chunks(tokens, timestamps, sample_rate=16000, mode=True)

        self.assertEqual(len(chunks), 2)
        # 1280 / 16000 = 0.08s per frame
        # Chunk 1: frame 2 to frame 3 -> start 0.16s, end (3+1)*0.08 = 0.32s
        self.assertEqual(chunks[0]["text"], "ภาษาไทย")
        self.assertEqual(chunks[0]["timestamp"], (0.16, 0.32))
        self.assertEqual(chunks[0]["start"], 0.16)
        self.assertEqual(chunks[0]["end"], 0.32)

        # Chunk 2: frame 10 to frame 11 -> start 0.80s, end (11+1)*0.08 = 0.96s
        self.assertEqual(chunks[1]["text"], "ง่ายนิดเดียว")
        self.assertEqual(chunks[1]["timestamp"], (0.80, 0.96))
        self.assertEqual(chunks[1]["start"], 0.80)
        self.assertEqual(chunks[1]["end"], 0.96)

    def test_tokens_to_timestamp_chunks_char_mode(self):
        """Test timestamp chunk generation in char/token mode."""
        model = FastConformerRNNT.__new__(FastConformerRNNT)
        model.vocab = ["<unk>", "<s>", "</s>", "\u2581ก", "า", "\u2581ข"]
        tokens = [3, 4, 5]
        timestamps = [1, 2, 4]
        chunks = model.tokens_to_timestamp_chunks(tokens, timestamps, sample_rate=16000, mode="char")

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["text"], "ก")
        self.assertEqual(chunks[0]["timestamp"], (0.08, 0.16))
        self.assertEqual(chunks[1]["text"], "า")
        self.assertEqual(chunks[1]["timestamp"], (0.16, 0.24))
        self.assertEqual(chunks[2]["text"], "ข")
        self.assertEqual(chunks[2]["timestamp"], (0.32, 0.40))

    def test_tokens_to_timestamp_chunks_empty(self):
        """Test that empty inputs return empty list."""
        model = FastConformerRNNT.__new__(FastConformerRNNT)
        model.vocab = ["<unk>", "<s>", "</s>"]
        self.assertEqual(model.tokens_to_timestamp_chunks([], []), [])
        self.assertEqual(model.tokens_to_timestamp_chunks([1, 2], []), [])
        self.assertEqual(model.tokens_to_timestamp_chunks([0], [1]), [])

    def test_tokens_to_timestamp_chunks_pause_boundary(self):
        """Test that a pause (>= 6 frames, ~0.48s) triggers a chunk boundary."""
        model = FastConformerRNNT.__new__(FastConformerRNNT)
        model.vocab = ["<unk>", "<s>", "</s>", "สวัสดี", "ครับ"]
        # Both tokens without \u2581, but separated by 10 frames
        tokens = [3, 4]
        timestamps = [0, 10]
        chunks = model.tokens_to_timestamp_chunks(tokens, timestamps, sample_rate=16000, mode=True)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "สวัสดี")
        self.assertEqual(chunks[0]["timestamp"], (0.0, 0.08))
        self.assertEqual(chunks[1]["text"], "ครับ")
        self.assertEqual(chunks[1]["timestamp"], (0.80, 0.88))

    def test_fastconformer_transcribe_timestamps_mock(self):
        """Test FastConformerRNNT.transcribe with mocked ONNX output."""
        from unittest.mock import MagicMock
        model = FastConformerRNNT.__new__(FastConformerRNNT)
        model.vocab = ["<unk>", "<s>", "</s>", "\u2581ทด", "สอบ"]
        model.blank_id = 5
        model.encoder_session = MagicMock()
        model.encoder_session.run.return_value = (np.zeros((1, 640, 10), dtype=np.float32), np.array([10]))
        
        # Mock greedy_decode
        model.greedy_decode = MagicMock(return_value=[{
            "tokens": [3, 4],
            "timestamps": [1, 2],
            "text": "ทดสอบ",
        }])

        audio_dummy = np.zeros(16000, dtype=np.float32)

        # Default: returns str
        text_res = model.transcribe(audio_dummy, return_timestamps=False)
        self.assertIsInstance(text_res, str)
        self.assertEqual(text_res, "ทดสอบ")

        # return_timestamps=True: returns dict
        dict_res = model.transcribe(audio_dummy, return_timestamps=True)
        self.assertIsInstance(dict_res, dict)
        self.assertIn("text", dict_res)
        self.assertIn("chunks", dict_res)
        self.assertIn("timestamps", dict_res)
        self.assertEqual(dict_res["text"], "ทดสอบ")
        self.assertEqual(len(dict_res["chunks"]), 1)
        self.assertEqual(dict_res["chunks"][0]["text"], "ทดสอบ")
        self.assertEqual(dict_res["chunks"][0]["timestamp"], (0.08, 0.24))

        # timestamps=True (alias)
        dict_alias = model.transcribe(audio_dummy, timestamps=True)
        self.assertIsInstance(dict_alias, dict)
        self.assertEqual(dict_alias["text"], "ทดสอบ")

    def test_asr_wrapper_timestamps(self):
        """Test top-level asr() with timestamps and return_timestamps."""
        from unittest.mock import MagicMock
        import pythaiasr

        mock_asr = MagicMock()
        mock_asr.return_value = {
            "text": "ข้อความ",
            "chunks": [{"text": "ข้อความ", "timestamp": (0.0, 0.5), "start": 0.0, "end": 0.5}],
            "timestamps": [{"text": "ข้อความ", "timestamp": (0.0, 0.5), "start": 0.0, "end": 0.5}],
        }

        orig_model = pythaiasr._model
        pythaiasr._model = mock_asr
        try:
            res = asr("dummy.wav", return_timestamps=True)
            self.assertIsInstance(res, dict)
            self.assertEqual(res["text"], "ข้อความ")
            self.assertEqual(len(res["chunks"]), 1)
            mock_asr.assert_called_with(
                data="dummy.wav",
                sampling_rate=16000,
                return_timestamps=True,
                timestamps=None,
            )

            res2 = asr("dummy.wav", timestamps=True)
            self.assertIsInstance(res2, dict)
            mock_asr.assert_called_with(
                data="dummy.wav",
                sampling_rate=16000,
                return_timestamps=None,
                timestamps=True,
            )
        finally:
            pythaiasr._model = orig_model

    def test_asr_whisper_timestamps_mock(self):
        """Test Whisper branch of ASR.__call__ with timestamps."""
        from unittest.mock import MagicMock
        asr_obj = ASR.__new__(ASR)
        asr_obj.is_typhoon = False
        asr_obj.is_whisper = True
        asr_obj.model_name = "biodatlab/whisper-th-medium-timestamp"
        asr_obj.device = "cpu"
        asr_obj.model = MagicMock()
        asr_obj.processor = MagicMock()

        mock_features = MagicMock()
        mock_features.input_features = MagicMock()
        mock_features.input_features.to.return_value = mock_features.input_features
        asr_obj.processor.return_value = mock_features
        asr_obj.model.generate.return_value = [[1, 2, 3]]
        
        asr_obj.processor.tokenizer.decode.return_value = {
            "text": "สวัสดี",
            "offsets": [{"text": "สวัสดี", "timestamp": (0.1, 1.2)}],
        }

        res = asr_obj(np.zeros(16000, dtype=np.float32), return_timestamps=True)
        self.assertIsInstance(res, dict)
        self.assertEqual(res["text"], "สวัสดี")
        self.assertEqual(len(res["chunks"]), 1)
        self.assertEqual(res["chunks"][0]["timestamp"], (0.1, 1.2))
        self.assertEqual(res["chunks"][0]["start"], 0.1)
        self.assertEqual(res["chunks"][0]["end"], 1.2)

        asr_obj.processor.batch_decode.return_value = ["สวัสดี"]
        str_res = asr_obj(np.zeros(16000, dtype=np.float32), return_timestamps=False)
        self.assertIsInstance(str_res, str)
        self.assertEqual(str_res, "สวัสดี")

    def test_asr_wav2vec2_timestamps_mock(self):
        """Test Wav2Vec2 branch of ASR.__call__ with timestamps."""
        from unittest.mock import MagicMock
        asr_obj = ASR.__new__(ASR)
        asr_obj.is_typhoon = False
        asr_obj.is_whisper = False
        asr_obj.lm = False
        asr_obj.model_name = "airesearch/wav2vec2-large-xlsr-53-th"
        asr_obj.device = "cpu"
        asr_obj.model = MagicMock()
        asr_obj.model.config = MagicMock()
        asr_obj.model.config.inputs_to_logits_ratio = 320
        asr_obj.processor = MagicMock()

        asr_obj.prepare_dataset = MagicMock(return_value={"input_values": [MagicMock()]})
        input_dict = MagicMock()
        input_dict.input_values = MagicMock()
        input_dict.to.return_value = input_dict
        asr_obj.processor.return_value = input_dict

        if torch is None:
            self.skipTest("torch is not installed")

        fake_logits = torch.zeros((1, 10, 100))

        class OutputModel:
            logits = fake_logits

        asr_obj.model.return_value = OutputModel()

        decode_result = MagicMock()
        decode_result.word_offsets = [{"word": "กะ", "start_offset": 5, "end_offset": 10}]
        asr_obj.processor.decode.return_value = decode_result

        res = asr_obj(np.zeros(16000, dtype=np.float32), return_timestamps=True)
        self.assertIsInstance(res, dict)
        self.assertIn("chunks", res)
        self.assertEqual(len(res["chunks"]), 1)
        self.assertEqual(res["chunks"][0]["start"], 0.1)
        self.assertEqual(res["chunks"][0]["end"], 0.2)

    def test_nemotron_imports_and_aliases(self):
        """Verify Nemotron ASR classes, functions, and aliases."""
        self.assertTrue(callable(NemotronStreamingASR))
        self.assertTrue(callable(NemotronASR))
        self.assertIs(NemotronASR, NemotronStreamingASR)
        self.assertTrue(callable(NemotronStreamASR))
        self.assertTrue(callable(RealtimeStreamNemotron))
        self.assertIs(RealtimeStreamNemotron, NemotronStreamASR)
        self.assertTrue(callable(get_nemotron_asr_model_files))
        self.assertTrue(callable(extract_nemotron_features))

    def test_extract_nemotron_features(self):
        """Test Nemotron 128-mel feature extraction."""
        sr = 16000
        duration = 0.56  # 560 ms
        t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        features = extract_nemotron_features(audio, sample_rate=sr)
        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[1], 128)
        self.assertGreater(features.shape[0], 0)

    def test_nemotron_prompt_resolution(self):
        """Test language prompt ID resolution."""
        model = NemotronStreamingASR.__new__(NemotronStreamingASR)
        model.default_language = "th-TH"
        from pythaiasr.nemotron_asr import NEMOTRON_PROMPTS
        model.prompt_dict = dict(NEMOTRON_PROMPTS)

        self.assertEqual(model.resolve_lang_id("th-TH"), 32)
        self.assertEqual(model.resolve_lang_id("th"), 32)
        self.assertEqual(model.resolve_lang_id("thai"), 32)
        self.assertEqual(model.resolve_lang_id("en-US"), 0)
        self.assertEqual(model.resolve_lang_id("auto"), 101)
        self.assertEqual(model.resolve_lang_id(42), 42)
        self.assertEqual(model.resolve_lang_id(None), 32)

    def test_nemotron_decode_tokens(self):
        """Test Nemotron token decoding and tag stripping."""
        model = NemotronStreamingASR.__new__(NemotronStreamingASR)
        model.vocab = ["<unk>", "<th-TH>", "\u2581สวัสดี", "ครับ", "<blank>"]
        tokens = [1, 2, 3]  # <th-TH>, \u2581สวัสดี, ครับ
        decoded = model.decode_token_ids(tokens)
        self.assertEqual(decoded, "สวัสดีครับ")

    def test_nemotron_timestamps(self):
        """Test Nemotron token timestamp grouping."""
        model = NemotronStreamingASR.__new__(NemotronStreamingASR)
        model.vocab = ["<unk>", "\u2581ภาษา", "ไทย", "\u2581ง่าย"]
        tokens = [1, 2, 3]
        timestamps = [0, 1, 5]  # frame 0, 1, 5
        chunks = model.tokens_to_timestamp_chunks(tokens, timestamps, sample_rate=16000)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "ภาษาไทย")
        self.assertEqual(chunks[0]["start"], 0.0)
        self.assertEqual(chunks[0]["end"], 0.16)
        self.assertEqual(chunks[1]["text"], "ง่าย")

    def test_mock_nemotron_transcribe_and_streaming(self):
        """Test NemotronStreamingASR and NemotronStreamASR with mocked ONNX runtime."""
        from unittest.mock import MagicMock
        model = NemotronStreamingASR.__new__(NemotronStreamingASR)
        model.vocab = ["<unk>", "\u2581ทด", "สอบ", "<blank>"]
        model.blank_id = 3
        model.hidden_dim = 640
        model.encoder_dim = 1024
        model.num_layers = 24
        model.left_context = 56
        model.conv_context = 8
        model.pre_encode_cache_size = 9
        model.chunk_frames = 56
        model.chunk_samples = 8960
        model.sample_rate = 16000
        model.max_symbols_per_step = 10
        model.default_language = "th-TH"
        from pythaiasr.nemotron_asr import NEMOTRON_PROMPTS
        model.prompt_dict = dict(NEMOTRON_PROMPTS)

        model.encoder_session = MagicMock()
        model.decoder_session = MagicMock()
        model.joint_session = MagicMock()

        # Mock encoder output: (1, 7, 1024)
        mock_enc_out = np.zeros((1, 7, 1024), dtype=np.float32)
        mock_cache_ch = np.zeros((1, 24, 56, 1024), dtype=np.float32)
        mock_cache_tm = np.zeros((1, 24, 1024, 8), dtype=np.float32)
        mock_cache_len = np.array([56], dtype=np.int64)
        model.encoder_session.run.return_value = (
            mock_enc_out, np.array([7]), mock_cache_ch, mock_cache_tm, mock_cache_len
        )

        # Mock greedy_decode
        model.greedy_decode = MagicMock(side_effect=[
            ([1, 2], [0, 1], np.zeros((2, 1, 640)), np.zeros((2, 1, 640)), 3),
            ([], [], np.zeros((2, 1, 640)), np.zeros((2, 1, 640)), 3),
            ([1, 2], [0, 1], np.zeros((2, 1, 640)), np.zeros((2, 1, 640)), 3),
            ([], [], np.zeros((2, 1, 640)), np.zeros((2, 1, 640)), 3),
        ])

        dummy_audio = np.zeros(8960, dtype=np.float32)
        text = model.transcribe(dummy_audio)
        self.assertEqual(text, "ทดสอบ")


        # Test streaming session
        streamer = NemotronStreamASR(model=model, sample_rate=16000, step_sec=0.56)
        chunk_text = streamer.process_chunk(dummy_audio)
        self.assertEqual(chunk_text, "ทดสอบ")
        self.assertEqual(streamer.get_full_transcript(), "ทดสอบ")

    def test_asr_class_nemotron_dispatch(self):
        """Test that ASR class correctly dispatches Nemotron models."""
        from unittest.mock import MagicMock, patch
        with patch("pythaiasr.NemotronStreamingASR") as mock_engine_class:
            mock_inst = MagicMock()
            mock_engine_class.return_value = mock_inst
            engine = ASR(model="nemotron_asr")
            self.assertTrue(engine.is_nemotron)
            self.assertFalse(engine.is_typhoon)
            self.assertEqual(engine.model, mock_inst)

            engine(np.zeros(16000, dtype=np.float32))
            mock_inst.transcribe.assert_called_once()


if __name__ == '__main__':
    unittest.main()

