import logging
import os

import httpx
from deepgram import DeepgramClient
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

def get_deepgram_client():
    return DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

STT_ATTEMPTS = 2  # one retry: dropped connections to Deepgram are usually transient
STT_TIMEOUT_SECONDS = 15  # transcription normally takes 1.5-2.5 s

def transcribe_audio_bytes(audio_bytes: bytes, mimetype: str = "audio/webm") -> str:
    """Transcribe raw audio bytes and return transcript."""
    client = get_deepgram_client()

    for attempt in range(1, STT_ATTEMPTS + 1):
        try:
            response = client.listen.v1.media.transcribe_file(
                request=audio_bytes,
                model="nova-2",
                language="en-US",
                smart_format=True,
                punctuate=True,
                request_options={"timeout_in_seconds": STT_TIMEOUT_SECONDS},
            )
            break
        except httpx.TransportError as e:  # network-level failure, not an API error
            if attempt == STT_ATTEMPTS:
                raise
            logger.warning("Deepgram connection failed (%s), retrying", e)

    transcript = response.results.channels[0].alternatives[0].transcript
    logger.debug("Transcript: %s", transcript)
    return transcript

def transcribe_audio_file(file_path: str) -> str:
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    return transcribe_audio_bytes(audio_bytes)
