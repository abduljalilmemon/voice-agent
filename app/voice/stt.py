import os
import asyncio
from deepgram import DeepgramClient, DeepgramClientOptions, PrerecordedOptions
from dotenv import load_dotenv

load_dotenv()

def get_deepgram_client():
    config = DeepgramClientOptions()
    return DeepgramClient(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        config=config
    )

def transcribe_audio_file(file_path: str) -> str:
    """Transcribe an audio file and return the transcript."""
    client = get_deepgram_client()

    with open(file_path, "rb") as audio:
        buffer_data = audio.read()

    payload = {"buffer": buffer_data}
    options = PrerecordedOptions(
        model="nova-2",
        language="en-US",
        smart_format=True,
        punctuate=True
    )

    response = client.listen.rest.v("1").transcribe_file(payload, options)
    transcript = response.results.channels[0].alternatives[0].transcript

    print(f"  [STT] -> '{transcript}'")
    return transcript

def transcribe_audio_bytes(audio_bytes: bytes, mimetype: str = "audio/webm") -> str:
    """Transcribe raw audio bytes and return the transcript."""
    client = get_deepgram_client()

    payload = {"buffer": audio_bytes, "mimetype": mimetype}
    options = PrerecordedOptions(
        model="nova-2",
        language="en-US",
        smart_format=True,
        punctuate=True
    )

    response = client.listen.rest.v("1").transcribe_file(payload, options)
    transcript = response.results.channels[0].alternatives[0].transcript

    print(f"  [STT] -> '{transcript}'")
    return transcript