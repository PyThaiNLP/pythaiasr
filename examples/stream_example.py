#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Live Audio Streaming with PyThaiASR

This example demonstrates how to use the stream_asr function to 
perform real-time speech recognition from microphone input.

Requirements:
    pip install pythaiasr[stream]

Usage:
    python stream_example.py
    
Press Ctrl+C to stop recording.
"""

from pythaiasr import stream_asr

def main():
    """
    Stream audio from microphone and print Thai transcriptions in real-time.
    """
    print("=" * 60)
    print("Live Audio Streaming Example")
    print("=" * 60)
    print()
    
    try:
        # Stream audio with 5-second chunks
        for transcription in stream_asr(
            model="airesearch/wav2vec2-large-xlsr-53-th",
            chunk_duration=5.0,
            device="cpu"  # Use "cuda" if you have GPU
        ):
            print(f"Transcription: {transcription}")
            print("-" * 60)
            
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
