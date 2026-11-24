import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge")

# Voice config per provider
VOICES = {
    "edge": "en-US-JennyNeural",
    "elevenlabs": "EXAVITQu4vr4xnSDxMaL",  # Sarah
    "openai": "nova"
}

async def text_to_speech_bytes(text: str) -> bytes:
    """Convert text to speech and return raw bytes. Never blocks the event loop."""
    provider = TTS_PROVIDER.lower()

    if provider == "edge":
        return await _edge_tts(text)
    elif provider == "elevenlabs":
        return await asyncio.to_thread(_elevenlabs_tts, text)
    elif provider == "openai":
        return await asyncio.to_thread(_openai_tts, text)
    else:
        raise ValueError(f"Unknown TTS provider: {provider}")

def text_to_speech(text: str, output_path: str = "output.mp3") -> str:
    """Convert text to speech and save to file (sync helper for scripts)."""
    audio_bytes = asyncio.run(text_to_speech_bytes(text))
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    print(f"  [TTS] -> saved to {output_path}")
    return output_path

# ── Edge TTS (free) ────────────────────────────────────────────

async def _edge_tts(text: str) -> bytes:
    import edge_tts
    voice = os.getenv("EDGE_VOICE", VOICES["edge"])
    communicate = edge_tts.Communicate(text, voice)
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    audio_bytes = b"".join(chunks)
    print(f"  [TTS] -> Edge | {len(audio_bytes)} bytes")
    return audio_bytes

# ── ElevenLabs TTS ─────────────────────────────────────────────

def _elevenlabs_tts(text: str) -> bytes:
    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", VOICES["elevenlabs"])
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.3,
            use_speaker_boost=True
        ),
        model_id="eleven_turbo_v2",
        output_format="mp3_44100_128"
    )
    audio_bytes = b""
    for chunk in audio:
        if isinstance(chunk, bytes):
            audio_bytes += chunk
    print(f"  [TTS] -> ElevenLabs | {len(audio_bytes)} bytes")
    return audio_bytes

# ── OpenAI TTS ─────────────────────────────────────────────────

def _openai_tts(text: str) -> bytes:
    from openai import OpenAI
    voice = os.getenv("OPENAI_VOICE", VOICES["openai"])
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    audio_bytes = response.content
    print(f"  [TTS] -> OpenAI | {len(audio_bytes)} bytes")
    return audio_bytes