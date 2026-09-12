#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Live Audio Streaming with PyThaiASR (Typhoon ASR)

This example demonstrates how to use the stream_asr function to 
perform real-time speech recognition from microphone input using 
Typhoon FastConformer RNN-T ONNX.

Requirements:
    pip install pythaiasr[stream]

Usage:
    python stream_example.py
    
Press Ctrl+C to stop recording.
"""

import os
import sys

# Ensure pythaiasr package in current repo can be imported directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythaiasr import stream_asr

def main():
    """
    Stream audio from microphone and print Thai transcriptions in real-time.
    """
    print("=" * 60)
    print("Live Audio Streaming Example (Typhoon ASR)")
    print("=" * 60)
    print()
    
    try:
        # Stream audio with Typhoon ASR (chunk duration: 0.48s)
        for transcription in stream_asr(
            model="typhoon_asr",
            chunk_duration=0.48,
            device="auto"  # "auto", "cpu", or "cuda"
        ):
            print(transcription, end=" ", flush=True)
            
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install PyAudio:")
        print("  pip install pythaiasr[stream]")
        print("\nOr manually:")
        print("  pip install pyaudio")
    except KeyboardInterrupt:
        print("\nStream stopped by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

