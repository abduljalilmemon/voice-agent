"""Shared fixtures. Tests run offline: fake LLMs, temp storage, no real email or API keys."""
import os
import tempfile
from pathlib import Path

# ── Environment -- must be set before anything imports `app` ───
# load_dotenv() never overrides variables that already exist, so these win over a local .env.
os.chdir(Path(__file__).resolve().parents[1])  # app reads data/ and frontend/ relative to the repo root
os.environ.update({
    "LLM_PROVIDER": "groq",
    "GROQ_API_KEY": "test-key-not-real",
    "STORAGE_DIR": tempfile.mkdtemp(prefix="voice-agent-tests-"),
    "ADMIN_API_KEY": "test-admin-key",
    "OWNER_EMAIL": "owner@example.com",
    "GMAIL_USER": "",          # empty => email sending is skipped unless a test fakes SMTP
    "GMAIL_APP_PASSWORD": "",
    "CORS_ORIGINS": "",
    "RATE_LIMIT_PER_MINUTE": "20",
})

import pytest  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from tests.fakes import FakeLLM  # noqa: E402


@pytest.fixture
def fake_llms(monkeypatch):
    """Install fake LLMs into the agent nodes: fake_llms(intents=[...], replies=[...], extractions=[...])."""
    import app.agent.nodes.booking_extractor as extractor_node
    import app.agent.nodes.intent as intent_node
    import app.agent.nodes.responder as responder_node

    def install(intents=(), replies=(), extractions=()):
        fakes = {"intent": FakeLLM(intents), "responder": FakeLLM(replies), "extractor": FakeLLM(extractions)}
        monkeypatch.setattr(intent_node, "llm", fakes["intent"])
        monkeypatch.setattr(responder_node, "llm", fakes["responder"])
        monkeypatch.setattr(extractor_node, "llm", fakes["extractor"])
        return fakes

    return install


@pytest.fixture
def saved_bookings(monkeypatch):
    """Capture bookings and emails instead of writing to SQLite / sending mail."""
    import app.agent.nodes.booking_saver as saver_node

    record = {"bookings": [], "emails": []}

    def fake_save(session_id, business, data):
        record["bookings"].append(data)
        return len(record["bookings"])

    monkeypatch.setattr(saver_node, "save_booking", fake_save)
    monkeypatch.setattr(saver_node, "send_booking_email", lambda *args: record["emails"].append(args) or True)
    return record


@pytest.fixture
def agent(monkeypatch):
    """The real agent graph with in-memory checkpoints and no knowledge-base lookups."""
    import app.agent.nodes.retriever as retriever_node
    from app.agent.graph import build_graph

    monkeypatch.setattr(retriever_node, "retrieve", lambda query: "")
    graph = build_graph(checkpointer=MemorySaver())

    def turn(text, session_id="test-session"):
        return graph.invoke(
            {"messages": [HumanMessage(content=text)], "intent": "", "context": "",
             "session_id": session_id, "new_booking_id": None},
            config={"configurable": {"thread_id": session_id}},
        )

    return turn


@pytest.fixture
def client(monkeypatch):
    """FastAPI test client with the agent replaced by an echo, and a fresh rate limiter."""
    from fastapi.testclient import TestClient

    import app.main as main
    from app.security import RateLimiter

    monkeypatch.setattr(main, "run_turn", lambda sid, text: {"messages": [AIMessage(content=f"echo: {text}")],
                                                             "intent": "FAQ"})
    monkeypatch.setattr(main, "turn_limiter", RateLimiter(max_events=1000))
    return TestClient(main.app)
