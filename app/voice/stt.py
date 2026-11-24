import os

from deepgram import DeepgramClient
from dotenv import load_dotenv

load_dotenv()

def get_deepgram_client():
    return DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

def transcribe_audio_bytes(audio_bytes: bytes, mimetype: str = "audio/webm") -> str:
    """Transcribe raw audio bytes and return transcript."""
    client = get_deepgram_client()

    response = client.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model="nova-2",
        language="en-US",
        smart_format=True,
        punctuate=True
    )

    transcript = response.results.channels[0].alternatives[0].transcript
    print(f"  [STT] -> '{transcript}'")
    return transcript

def transcribe_audio_file(file_path: str) -> str:
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    return transcribe_audio_bytes(audio_bytes)