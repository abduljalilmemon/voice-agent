import json

import pytest
from starlette.websockets import WebSocketDisconnect

import app.main as main
import app.security as security
from app.security import MAX_MESSAGE_CHARS, RateLimiter

ADMIN = {"X-API-Key": "test-admin-key"}
OWN_ORIGIN = {"Origin": "http://testserver"}
FOREIGN_ORIGIN = {"Origin": "https://evil.example"}


# ── Admin endpoint ─────────────────────────────────────────────

@pytest.mark.parametrize("headers, status", [({}, 401), ({"X-API-Key": "wrong"}, 401), (ADMIN, 200)])
def test_bookings_requires_admin_key(client, headers, status):
    assert client.get("/bookings", headers=headers).status_code == status


def test_bookings_disabled_when_no_admin_key_configured(client, monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY")
    assert client.get("/bookings", headers=ADMIN).status_code == 503


# ── CORS ───────────────────────────────────────────────────────

def test_cors_does_not_allow_unknown_origins(client):
    response = client.options("/chat", headers={**FOREIGN_ORIGIN, "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in response.headers


# ── /chat validation ───────────────────────────────────────────

@pytest.mark.parametrize("message", ["", "x" * (MAX_MESSAGE_CHARS + 1)])
def test_chat_rejects_empty_or_oversized_messages(client, message):
    assert client.post("/chat", json={"message": message}).status_code == 422


def test_chat_works_and_assigns_a_session_id(client):
    body = client.post("/chat", json={"message": "hi"}).json()
    assert body["reply"] == "echo: hi"
    assert body["session_id"]


# ── WebSocket origin + session id checks ───────────────────────

def _rejected(client, path, headers):
    try:
        with client.websocket_connect(path, headers=headers) as ws:
            ws.send_text(json.dumps({"message": "hi"}))
            ws.receive_text()
        return False
    except WebSocketDisconnect as exc:
        return exc.code == 1008


@pytest.mark.parametrize("path", ["/ws/abc", "/ws/voice/abc"])
def test_websockets_reject_other_sites(client, path):
    assert _rejected(client, path, FOREIGN_ORIGIN)


def test_websocket_rejects_oversized_session_id(client):
    assert _rejected(client, "/ws/" + "a" * 101, OWN_ORIGIN)


def test_text_websocket_from_own_frontend(client):
    with client.websocket_connect("/ws/s1", headers=OWN_ORIGIN) as ws:
        ws.send_text(json.dumps({"message": "hi"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        assert json.loads(ws.receive_text())["reply"] == "echo: hi"


def test_text_websocket_survives_bad_input(client):
    with client.websocket_connect("/ws/s1") as ws:
        ws.send_text("not json")
        assert json.loads(ws.receive_text())["type"] == "error"
        ws.send_text(json.dumps({"message": "x" * (MAX_MESSAGE_CHARS + 1)}))
        assert "too long" in json.loads(ws.receive_text())["message"]
        ws.send_text(json.dumps({"message": "still connected?"}))  # connection is still usable
        ws.receive_text()
        assert json.loads(ws.receive_text())["reply"] == "echo: still connected?"


def test_voice_websocket_rejects_oversized_audio(client):
    with client.websocket_connect("/ws/voice/s2") as ws:
        ws.send_bytes(b"\0" * (security.MAX_AUDIO_BYTES + 1))
        assert "too long" in json.loads(ws.receive_text())["message"]


# ── Rate limiting ──────────────────────────────────────────────

def test_rate_limit_applies_across_rest_and_websockets(client, monkeypatch):
    monkeypatch.setattr(main, "turn_limiter", RateLimiter(max_events=3))
    codes = [client.post("/chat", json={"message": "hi"}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    with client.websocket_connect("/ws/s3") as ws:
        ws.send_text(json.dumps({"message": "hi"}))
        assert "too quickly" in json.loads(ws.receive_text())["message"]


def test_rate_limiter_window_slides(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: now[0])
    limiter = RateLimiter(max_events=2, window_seconds=60)

    assert limiter.allow("ip") and limiter.allow("ip")
    assert not limiter.allow("ip")
    assert limiter.allow("other-ip")  # limits are per client
    now[0] += 61
    assert limiter.allow("ip")        # old events expired


# ── Public business details ────────────────────────────────────

def test_business_endpoint_serves_the_configured_name(client, monkeypatch):
    monkeypatch.setitem(main.BIZ_CONFIG, "business_name", "Smile Studio")
    assert client.get("/business").json() == {"name": "Smile Studio"}


def test_business_endpoint_exposes_only_public_details(client):
    assert set(client.get("/business").json()) == {"name"}
