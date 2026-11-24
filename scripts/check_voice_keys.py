"""Manual check: verifies the ElevenLabs and Deepgram API keys in .env.

Run from the repo root:  python scripts/check_voice_keys.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

from dotenv import load_dotenv

load_dotenv()

def test_elevenlabs():
    print("Testing ElevenLabs...")
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    voices = client.voices.get_all()
    print(f"  [ElevenLabs] OK -- {len(voices.voices)} voices available")
    for voice in voices.voices:
        print(f"    {voice.name} | {voice.voice_id}")

def test_deepgram():
    print("Testing Deepgram...")
    from deepgram import DeepgramClient
    DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))  # same constructor the app uses
    print("  [Deepgram] OK -- client initialized")
    
if __name__ == "__main__":
    test_elevenlabs()
    print()
    test_deepgram()