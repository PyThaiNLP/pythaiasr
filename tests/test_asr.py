# -*- coding: utf-8 -*-

import unittest
import torchaudio
from pythaiasr import asr, ASR
import os

file = os.path.join(".", "tests", "common_voice_th_25686161.wav")

class TestKhaveePackage(unittest.TestCase):
    def test_asr(self):
        self.assertIsNotNone(asr(file, device="cpu"))
    def test_asr_array(self):
        speech_array, sampling_rate = torchaudio.load(file)
        self.assertIsNotNone(asr(speech_array[0].numpy(), device="cpu", sampling_rate=sampling_rate))
    
    def test_typhoon_model_in_supported_list(self):
        """Test that typhoon-asr-realtime is in the supported models list"""
        model = ASR()
        self.assertIn("scb10x/typhoon-asr-realtime", model.support_model)
    
    @unittest.skipUnless(
        os.environ.get('TEST_TYPHOON_ASR', 'false').lower() == 'true',
        "Skipping Typhoon ASR test - set TEST_TYPHOON_ASR=true to run"
    )
    def test_typhoon_asr(self):
        """Test typhoon-asr-realtime model (requires nemo-toolkit)"""
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            self.skipTest("nemo-toolkit not installed")
        
        # Test that model initialization doesn't fail
        result = asr(file, model="scb10x/typhoon-asr-realtime", device="cpu")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
    
    @unittest.skipUnless(
        os.environ.get('TEST_TYPHOON_ASR', 'false').lower() == 'true',
        "Skipping Typhoon ASR test - set TEST_TYPHOON_ASR=true to run"
    )
    def test_typhoon_asr_array(self):
        """Test typhoon-asr-realtime with numpy array input"""
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            self.skipTest("nemo-toolkit not installed")
        
        speech_array, sampling_rate = torchaudio.load(file)
        result = asr(speech_array[0].numpy(), model="scb10x/typhoon-asr-realtime", 
                     device="cpu", sampling_rate=sampling_rate)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
    
    def test_typhoon_import_error(self):
        """Test that appropriate error is raised when nemo-toolkit is not installed"""
        # Mock the import by temporarily modifying sys.modules if nemo is installed
        import sys
        nemo_backup = sys.modules.get('nemo.collections.asr', None)
        
        try:
            # Temporarily remove nemo from sys.modules if it exists
            if 'nemo.collections.asr' in sys.modules:
                del sys.modules['nemo.collections.asr']
            if 'nemo.collections' in sys.modules:
                del sys.modules['nemo.collections']
            if 'nemo' in sys.modules:
                del sys.modules['nemo']
            
            # Try to create ASR with typhoon model
            with self.assertRaises(ImportError) as context:
                ASR(model="scb10x/typhoon-asr-realtime", device="cpu")
            
            self.assertIn("nemo-toolkit", str(context.exception))
            self.assertIn("pythaiasr[typhoon]", str(context.exception))
        finally:
            # Restore nemo if it was available
            if nemo_backup:
                sys.modules['nemo.collections.asr'] = nemo_backup

