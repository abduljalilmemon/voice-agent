# test_voice_keys.py
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
    deepgram = DeepgramClient(access_token=os.getenv("DEEPGRAM_API_KEY"))
    print(f"  [Deepgram] OK -- client initialized")
    
if __name__ == "__main__":
    test_elevenlabs()
    print()
    test_deepgram()