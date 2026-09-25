#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Speech Diarization and ASR Diarization with PyThaiASR

Usage:
    python diarize_example.py [audio_file.wav]
"""

import os
import sys

# Ensure pythaiasr package in current repo can be imported directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythaiasr import diarize, asr_diarize, segments_to_rttm


def main():
    if len(sys.argv) > 1:
        test_audio = sys.argv[1]
    else:
        test_audio = os.path.join(os.path.dirname(__file__), "..", "tests", "test-diarize.wav")

    if not os.path.exists(test_audio):
        print(f"Audio file '{test_audio}' not found. Please provide an audio file path.")
        sys.exit(1)

    print("=" * 60)
    print("1. Speech Diarization (defaults to Nemotron-3 INT8 ONNX)")
    print("=" * 60)
    print(f"Processing: {test_audio} ...")
    segments = diarize(test_audio, device="auto")

    if not segments:
        print("No speech segments detected.")
    else:
        for seg in segments:
            print(f"[{seg['start']:6.2f}s -> {seg['end']:6.2f}s] {seg['speaker']}")

        print("\nNIST RTTM Format:")
        print(segments_to_rttm(segments, uri=os.path.basename(test_audio)))

    print("=" * 60)
    print("2. Diarization + Speech Recognition (ASR Diarize)")
    print("=" * 60)
    print(f"Transcribing turns with Typhoon ASR: {test_audio} ...")
    turns = asr_diarize(
        test_audio,
        asr_model="typhoon_asr",
        device="auto",
    )

    if not turns:
        print("No speech turns found.")
    else:
        for turn in turns:
            print(f"[{turn['start']:6.2f}s -> {turn['end']:6.2f}s] {turn['speaker']}: {turn['text']}")


if __name__ == "__main__":
    main()


