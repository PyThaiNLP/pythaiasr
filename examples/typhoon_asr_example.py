#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Typhoon ASR (Offline and Realtime Streaming) with PyThaiASR

Requirements:
    pip install pythaiasr[typhoon]

Usage:
    python typhoon_asr_example.py
"""

import os
import sys

# Ensure pythaiasr package in current repo can be imported directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythaiasr import asr, stream_from_file, FastConformerRNNT, RealtimeStreamASR

def main():
    test_audio = os.path.join(os.path.dirname(__file__), "..", "tests", "test.wav")
    if not os.path.exists(test_audio):
        test_audio = "test.wav"

    print("=" * 60)
    print("1. Typhoon Offline ASR")
    print("=" * 60)
    if os.path.exists(test_audio):
        text = asr(test_audio, model="typhoon_asr", device="auto")
        print("Transcription:", text)
    else:
        print(f"Audio file {test_audio} not found. Pass an audio path to asr().")

    print("\n" + "=" * 60)
    print("2. Typhoon Realtime Streaming Simulation (Chunk-by-Chunk)")
    print("=" * 60)
    if os.path.exists(test_audio):
        model = FastConformerRNNT(device="auto")
        streamer = RealtimeStreamASR(model=model, step_sec=0.48)
        stream_from_file(streamer, test_audio, chunk_sec=0.48, simulate_realtime=True)


if __name__ == "__main__":
    main()
