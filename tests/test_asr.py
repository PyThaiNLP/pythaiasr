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
