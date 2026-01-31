# -*- coding: utf-8 -*-

import unittest
import torchaudio
from pythaiasr import asr
import os

file = os.path.join(".", "tests", "common_voice_th_25686161.wav")

class TestKhaveePackage(unittest.TestCase):
    def test_asr(self):
        self.assertIsNotNone(asr(file, device="cpu"))
    def test_asr_array(self):
        speech_array, sampling_rate = torchaudio.load(file)
        self.assertIsNotNone(asr(speech_array[0].numpy(), device="cpu", sampling_rate=sampling_rate))
    def test_whisper_small(self):
        self.assertIsNotNone(asr(file, model="biodatlab/whisper-small-th-combined", device="cpu"))
    def test_whisper_medium(self):
        self.assertIsNotNone(asr(file, model="biodatlab/whisper-th-medium-combined", device="cpu"))
    def test_whisper_large(self):
        self.assertIsNotNone(asr(file, model="biodatlab/whisper-th-large-combined", device="cpu"))
    def test_whisper_array(self):
        speech_array, sampling_rate = torchaudio.load(file)
        self.assertIsNotNone(asr(speech_array[0].numpy(), model="biodatlab/whisper-small-th-combined", device="cpu", sampling_rate=sampling_rate))
    
    def test_stream_asr_import(self):
        """Test that stream_asr can be imported"""
        from pythaiasr import stream_asr
        self.assertTrue(callable(stream_asr))
    
    def test_stream_asr_without_pyaudio(self):
        """Test that stream_asr raises ImportError when pyaudio is not available"""
        # Temporarily hide pyaudio if it exists
        import sys
        pyaudio_backup = sys.modules.get('pyaudio')
        if 'pyaudio' in sys.modules:
            del sys.modules['pyaudio']
        
        try:
            from pythaiasr import stream_asr
            # Try to call the generator (it should raise ImportError)
            gen = stream_asr(device="cpu")
            with self.assertRaises(ImportError) as context:
                next(gen)  # This should trigger the pyaudio import
            self.assertIn("pyaudio is required", str(context.exception))
        finally:
            # Restore pyaudio if it was available
            if pyaudio_backup is not None:
                sys.modules['pyaudio'] = pyaudio_backup

