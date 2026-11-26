"""Failures in one step must reach the user as a message and never kill the connection."""
import io
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from starlette.websockets import WebSocketDisconnect

import app.main as main
from app.logging_setup import setup_logging
from app.voice import stt


def boom(*args, **kwargs):
    raise RuntimeError("simulated provider outage")


async def fake_tts(text):
    return b"mp3-bytes"


def next_of_type(ws, wanted):
    """Skip status/typing updates and return the next message of a given type."""
    while True:
        message = json.loads(ws.receive_text())
        if message["type"] in (wanted, "error"):
            return message


# ── /chat ──────────────────────────────────────────────────────

def test_chat_returns_503_when_the_agent_fails(client, monkeypatch):
    monkeypatch.setattr(main, "run_turn", boom)
    response = client.post("/chat", json={"message": "hi"})
    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]


# ── Text WebSocket ─────────────────────────────────────────────

def test_text_websocket_reports_agent_failure_and_stays_open(client, monkeypatch):
    with client.websocket_connect("/ws/s1") as ws:
        monkeypatch.setattr(main, "run_turn", boom)
        ws.send_text(json.dumps({"message": "hi"}))
        assert "temporarily unavailable" in next_of_type(ws, "message")["message"]

        monkeypatch.setattr(main, "run_turn", lambda sid, text: {"messages": [SimpleNamespace(content="back")],
                                                                 "intent": "FAQ"})
        ws.send_text(json.dumps({"message": "hi again"}))
        assert next_of_type(ws, "message")["reply"] == "back"


def test_text_websocket_closes_cleanly_on_unexpected_frame(client):
    with client.websocket_connect("/ws/s1") as ws:
        ws.send_bytes(b"\x00binary")
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 1011


# ── Voice WebSocket ────────────────────────────────────────────

@pytest.fixture
def voice(client, monkeypatch):
    monkeypatch.setattr(main, "transcribe_audio_bytes", lambda data, mimetype: "what are your hours?")
    monkeypatch.setattr(main, "text_to_speech_bytes", fake_tts)
    return client


def test_voice_happy_path(voice):
    with voice.websocket_connect("/ws/voice/v1") as ws:
        ws.send_bytes(b"audio")
        assert next_of_type(ws, "transcript")["text"] == "what are your hours?"
        assert next_of_type(ws, "reply")["text"] == "echo: what are your hours?"
        assert next_of_type(ws, "audio")["format"] == "mp3"


def test_voice_reports_transcription_failure_and_stays_open(voice, monkeypatch):
    with voice.websocket_connect("/ws/voice/v1") as ws:
        monkeypatch.setattr(main, "transcribe_audio_bytes", boom)
        ws.send_bytes(b"audio")
        assert "couldn't process that audio" in next_of_type(ws, "transcript")["message"]

        monkeypatch.setattr(main, "transcribe_audio_bytes", lambda data, mimetype: "hello")
        ws.send_bytes(b"audio")
        assert next_of_type(ws, "transcript")["text"] == "hello"


def test_voice_reports_agent_failure(voice, monkeypatch):
    monkeypatch.setattr(main, "run_turn", boom)
    with voice.websocket_connect("/ws/voice/v1") as ws:
        ws.send_bytes(b"audio")
        assert "temporarily unavailable" in next_of_type(ws, "reply")["message"]


def test_voice_still_delivers_text_reply_when_tts_fails(voice, monkeypatch):
    async def tts_down(text):
        raise RuntimeError("tts outage")

    monkeypatch.setattr(main, "text_to_speech_bytes", tts_down)
    with voice.websocket_connect("/ws/voice/v1") as ws:
        ws.send_bytes(b"audio")
        assert next_of_type(ws, "reply")["text"] == "echo: what are your hours?"
        assert "Voice playback is unavailable" in next_of_type(ws, "audio")["message"]


# ── Deepgram retry ─────────────────────────────────────────────

def _deepgram(monkeypatch, outcomes):
    """Fake Deepgram client: each call pops the next outcome (an exception or a transcript)."""
    calls = []

    def transcribe_file(**kwargs):
        calls.append(kwargs)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        alt = SimpleNamespace(transcript=outcome)
        return SimpleNamespace(results=SimpleNamespace(channels=[SimpleNamespace(alternatives=[alt])]))

    client = SimpleNamespace(listen=SimpleNamespace(v1=SimpleNamespace(media=SimpleNamespace(
        transcribe_file=transcribe_file))))
    monkeypatch.setattr(stt, "get_deepgram_client", lambda: client)
    return calls


def test_stt_retries_once_after_a_dropped_connection(monkeypatch):
    calls = _deepgram(monkeypatch, [httpx.ConnectError("connection reset"), "hello"])
    assert stt.transcribe_audio_bytes(b"audio") == "hello"
    assert len(calls) == 2


def test_stt_gives_up_after_repeated_network_failures(monkeypatch):
    _deepgram(monkeypatch, [httpx.ConnectError("down"), httpx.ConnectError("still down")])
    with pytest.raises(httpx.ConnectError):
        stt.transcribe_audio_bytes(b"audio")


def test_stt_does_not_retry_api_errors(monkeypatch):
    calls = _deepgram(monkeypatch, [ValueError("bad audio"), "unused"])
    with pytest.raises(ValueError):
        stt.transcribe_audio_bytes(b"audio")
    assert len(calls) == 1


# ── Logging ────────────────────────────────────────────────────

def test_logging_emoji_to_a_legacy_encoded_stream_does_not_crash():
    # what Windows does when output is redirected to a file: cp1252 can't encode emoji
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    setup_logging(stream)
    try:
        logging.getLogger("app.test").info("Reply: Hi there! \U0001f44b")
        stream.flush()
        assert b"Hi there! \\U0001f44b" in raw.getvalue()
    finally:
        setup_logging()  # restore the normal handler for other tests


# ── One turn at a time per conversation ────────────────────────

def _concurrency_probe(monkeypatch):
    import threading
    import time

    state = {"running": {}, "peak": {}}
    guard = threading.Lock()

    def slow_turn(session_id, text):
        with guard:
            state["running"][session_id] = state["running"].get(session_id, 0) + 1
            total = sum(state["running"].values())
            state["peak"][session_id] = max(state["peak"].get(session_id, 0), state["running"][session_id])
            state["peak"]["all"] = max(state["peak"].get("all", 0), total)
        time.sleep(0.2)
        with guard:
            state["running"][session_id] -= 1
        return {}

    monkeypatch.setattr(main, "_invoke_agent", slow_turn)
    return state


def _run_together(calls):
    import threading

    threads = [threading.Thread(target=main.run_turn, args=args) for args in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_turns_for_the_same_session_never_overlap(monkeypatch):
    state = _concurrency_probe(monkeypatch)
    _run_together([("same", "a"), ("same", "b"), ("same", "c")])
    assert state["peak"]["same"] == 1


def test_different_sessions_still_run_in_parallel(monkeypatch):
    state = _concurrency_probe(monkeypatch)
    _run_together([("caller-1", "a"), ("caller-2", "b")])
    assert state["peak"]["all"] == 2


# ── Provider rate limits ───────────────────────────────────────

class RateLimitError(Exception):
    """Named like Groq's / Anthropic's / OpenAI's 429 errors."""


def _rate_limited(*args, **kwargs):
    raise RateLimitError("429 Too Many Requests")


def test_rate_limit_gives_a_busy_message_over_rest(client, monkeypatch):
    monkeypatch.setattr(main, "run_turn", _rate_limited)
    response = client.post("/chat", json={"message": "hi"})
    assert response.status_code == 503 and "a lot of requests" in response.json()["detail"]


def test_rate_limit_gives_a_busy_message_over_websocket(client, monkeypatch):
    monkeypatch.setattr(main, "run_turn", _rate_limited)
    with client.websocket_connect("/ws/s1") as ws:
        ws.send_text(json.dumps({"message": "hi"}))
        assert "a lot of requests" in next_of_type(ws, "message")["message"]


def test_groq_calls_retry_rate_limits():
    from groq import RateLimitError as GroqRateLimitError

    from app.llm import get_llm

    assert GroqRateLimitError in get_llm().retry_exception_types
