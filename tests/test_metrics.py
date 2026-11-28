import json

import pytest

import app.main as main
from app.metrics import LatencyMetrics, metrics, percentile
from tests.fakes import understood

ADMIN = {"X-API-Key": "test-admin-key"}


# ── the maths ──────────────────────────────────────────────────

def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 101)]   # 1..100
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile([7.0], 95) == 7


def test_summary_reports_milliseconds():
    m = LatencyMetrics()
    for seconds in (0.1, 0.2, 0.3, 0.4, 1.0):
        m.record("stt", seconds)
    assert m.summary() == {"stt": {"count": 5, "p50_ms": 300, "p95_ms": 1000, "max_ms": 1000}}


def test_only_the_most_recent_samples_are_kept():
    m = LatencyMetrics(window=3)
    for seconds in (9.0, 9.0, 0.1, 0.1, 0.1):
        m.record("tts", seconds)
    assert m.summary()["tts"] == {"count": 3, "p50_ms": 100, "p95_ms": 100, "max_ms": 100}


def test_timer_records_even_when_the_call_fails():
    m = LatencyMetrics()
    timings = {}
    with pytest.raises(RuntimeError), m.timer("agent", timings):
        raise RuntimeError("provider down")
    assert m.summary()["agent"]["count"] == 1
    assert "agent" in timings   # a timeout is latency the caller waited through


# ── the endpoint ───────────────────────────────────────────────

def test_metrics_requires_the_admin_key(client):
    assert client.get("/metrics").status_code == 401


def test_rest_turn_is_timed(client):
    client.post("/chat", json={"message": "hi"})
    assert {"agent", "turn.rest"} <= set(client.get("/metrics", headers=ADMIN).json()["stages"])


def test_text_websocket_turn_is_timed(client):
    with client.websocket_connect("/ws/s1") as ws:
        ws.send_text(json.dumps({"message": "hi"}))
        ws.receive_text()
        ws.receive_text()
    assert {"agent", "turn.text"} <= set(metrics.summary())


async def _fake_tts(text):
    return b"mp3"


def test_voice_turn_times_every_stage(client, monkeypatch):
    monkeypatch.setattr(main, "transcribe_audio_bytes", lambda data, mimetype: "hello")
    monkeypatch.setattr(main, "text_to_speech_bytes", _fake_tts)
    with client.websocket_connect("/ws/voice/v1") as ws:
        ws.send_bytes(b"audio")
        while json.loads(ws.receive_text())["type"] != "audio":
            pass
    assert {"stt", "agent", "tts", "turn.voice"} <= set(metrics.summary())


def test_a_failed_turn_records_the_stage_but_no_total(client, monkeypatch):
    def stt_down(data, mimetype):
        raise RuntimeError("deepgram down")

    monkeypatch.setattr(main, "transcribe_audio_bytes", stt_down)
    with client.websocket_connect("/ws/voice/v1") as ws:
        ws.send_bytes(b"audio")
        while json.loads(ws.receive_text())["type"] != "error":
            pass
    stages = set(metrics.summary())
    assert "stt" in stages and "turn.voice" not in stages


# ── agent steps ────────────────────────────────────────────────

def test_each_agent_step_is_timed(fake_llms, saved_bookings, agent):
    fake_llms(turns=[understood("BOOK", name="A", date="2026-11-30", time="09:00", phone="03001234567")],
              replies=["booked"])
    agent("book A Monday 9am, 03001234567")
    assert {"agent.understand", "agent.knowledge", "agent.save_booking", "agent.reply"} <= set(metrics.summary())


# ── startup warm-up ────────────────────────────────────────────

def test_startup_loads_the_knowledge_base(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "retrieve", lambda query: calls.append(query) or "")
    main.warm_up_knowledge_base()
    assert calls == ["warm up"]


def test_failed_warm_up_does_not_stop_the_server(monkeypatch):
    def broken(query):
        raise RuntimeError("chroma missing")

    monkeypatch.setattr(main, "retrieve", broken)
    main.warm_up_knowledge_base()   # logs a warning, doesn't raise
