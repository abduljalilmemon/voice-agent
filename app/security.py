"""Request guards: admin auth, WebSocket origin checks, input limits, rate limiting."""
import os
import secrets
import threading
import time
from collections import deque
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import Header, HTTPException, WebSocket

load_dotenv()

# ── Input limits ───────────────────────────────────────────────

MAX_MESSAGE_CHARS = 1000
MAX_AUDIO_BYTES = 5 * 1024 * 1024  # ~5 MB of webm/opus is several minutes of speech
MAX_SESSION_ID_CHARS = 100

# ── Admin auth ─────────────────────────────────────────────────

def require_admin(x_api_key: str | None = Header(default=None)) -> None:
    """Protect admin endpoints with the ADMIN_API_KEY from .env (sent as the X-API-Key header)."""
    expected = os.getenv("ADMIN_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Admin API disabled: set ADMIN_API_KEY in .env")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

# ── Allowed origins (CORS + WebSockets) ────────────────────────

# Comma-separated extra origins allowed to call the API from a browser, e.g. "https://myclinic.com".
# The bundled frontend is served from the same origin and needs no entry.
ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

def websocket_origin_allowed(websocket: WebSocket) -> bool:
    """CORS does not apply to WebSockets, so check the browser's Origin header ourselves.

    Without this, any website a visitor opens could talk to the agent from their browser.
    Non-browser clients send no Origin header and are allowed (they are rate limited).
    """
    origin = websocket.headers.get("origin")
    if origin is None:
        return True
    if origin.rstrip("/") in ALLOWED_ORIGINS:
        return True
    return urlparse(origin).netloc == websocket.headers.get("host")

# ── Rate limiting ──────────────────────────────────────────────

class RateLimiter:
    """Sliding-window limit of `max_events` per `window_seconds` per key (client IP).

    In-memory, so it applies per server process -- enough for a single instance.
    """

    def __init__(self, max_events: int, window_seconds: float = 60.0):
        self.max_events = max_events
        self.window = window_seconds
        self._events: dict[str, deque] = {}
        self._lock = threading.Lock()  # /chat runs in FastAPI's thread pool

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and now - events[0] > self.window:
                events.popleft()
            if len(events) >= self.max_events:
                return False
            events.append(now)
            # drop idle clients so the dict doesn't grow forever
            if len(self._events) > 10_000:
                self._events = {k: v for k, v in self._events.items() if v and now - v[-1] <= self.window}
            return True

# every agent turn (text or voice) costs LLM calls, so they share one budget per client
turn_limiter = RateLimiter(max_events=int(os.getenv("RATE_LIMIT_PER_MINUTE", "20")))

def client_key(host: str | None) -> str:
    return host or "unknown"
