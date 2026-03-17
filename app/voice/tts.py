import os
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from dotenv import load_dotenv

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Sarah -- Mature, Reassuring, Confident
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"

def text_to_speech(text: str, output_path: str = "output.mp3") -> str:
    """Convert text to speech and save to file. Returns file path."""
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.3,
            use_speaker_boost=True
        ),
        model_id="eleven_turbo_v2",
        output_format="mp3_44100_128"
    )

    with open(output_path, "wb") as f:
        for chunk in audio:
            if isinstance(chunk, bytes):
                f.write(chunk)

    print(f"  [TTS] -> saved to {output_path}")
    return output_path

def text_to_speech_bytes(text: str) -> bytes:
    """Convert text to speech and return raw bytes for streaming."""
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
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

    print(f"  [TTS] -> generated {len(audio_bytes)} bytes")
    return audio_bytes