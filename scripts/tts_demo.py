"""Manual check: generates test_output.mp3 with the configured TTS provider.

Run from the repo root:  python scripts/tts_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

from dotenv import load_dotenv

from app.voice.tts import text_to_speech

load_dotenv()

def test_tts():
    print("Testing TTS...")
    path = text_to_speech(
        "Hello, welcome to City Dental Clinic. How can I assist you today?",
        output_path="test_output.mp3"
    )
    size = os.path.getsize(path)
    print(f"  [TTS] OK -- file size: {size} bytes")
    print("  Open test_output.mp3 to hear the voice")

if __name__ == "__main__":
    test_tts()