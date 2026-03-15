import os
from dotenv import load_dotenv
from app.voice.tts import text_to_speech, text_to_speech_bytes

load_dotenv()

def test_tts():
    print("Testing TTS...")
    path = text_to_speech(
        "Hello, welcome to City Dental Clinic. How can I assist you today?",
        output_path="test_output.mp3"
    )
    size = os.path.getsize(path)
    print(f"  [TTS] OK — file size: {size} bytes")
    print(f"  Open test_output.mp3 to hear the voice")

if __name__ == "__main__":
    test_tts()