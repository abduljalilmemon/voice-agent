from langchain_core.messages import HumanMessage

import app.agent.nodes.responder as responder
import app.knowledge.retriever as retriever
from app.config import BIZ_CONFIG
from tests.fakes import FakeLLM, understood

PHONE = BIZ_CONFIG["phone"]


def _reply(monkeypatch, state, llm_output="ok"):
    fake = FakeLLM([llm_output])
    monkeypatch.setattr(responder, "llm", fake)
    result = responder.generate_response({"messages": [HumanMessage(content="hi")], **state})
    system_prompt = fake.calls[0][0].content if fake.calls else None
    return result["messages"][0].content, system_prompt


# ── Prompt contents ────────────────────────────────────────────

def test_prompt_knows_the_business_and_today(monkeypatch):
    _, prompt = _reply(monkeypatch, {"intent": "GREETING"})
    assert BIZ_CONFIG["business_name"] in prompt and PHONE in prompt and BIZ_CONFIG["address"] in prompt
    assert "Today is " in prompt


def test_prompt_includes_retrieved_context(monkeypatch):
    _, prompt = _reply(monkeypatch, {"intent": "FAQ", "context": "Teeth whitening: PKR 8,000"})
    assert "Teeth whitening: PKR 8,000" in prompt


def test_static_replies_make_no_llm_call_and_escalation_is_honest(monkeypatch):
    reply, prompt = _reply(monkeypatch, {"intent": "ESCALATE"})
    assert prompt is None
    assert PHONE in reply and "hold" not in reply.lower()


# ── Booking state is told to the model, not guessed ────────────

def test_booking_in_progress_asks_for_next_missing_item_and_forbids_confirming(monkeypatch):
    _, prompt = _reply(monkeypatch, {"intent": "COLLECT", "booking_data": {"name": "Sara", "date": "Friday"}})
    assert "name: Sara" in prompt
    assert "Ask only for the next missing item: the preferred time" in prompt
    assert "NOT booked yet" in prompt


def test_missing_date_and_time_are_asked_together(monkeypatch):
    _, prompt = _reply(monkeypatch, {"intent": "COLLECT", "booking_data": {"name": "Sara"}})
    assert "next missing item: the preferred date and time" in prompt


def test_just_saved_booking_is_confirmed(monkeypatch):
    data = {"name": "Sara", "date": "Friday", "time": "10am", "phone": "0300"}
    _, prompt = _reply(monkeypatch, {"intent": "COLLECT", "booking_data": data, "new_booking_id": 7})
    assert "has just been saved" in prompt and "NOT booked" not in prompt


def test_existing_booking_points_changes_to_the_phone(monkeypatch):
    _, prompt = _reply(monkeypatch, {"intent": "COLLECT", "booking_data": {"name": "Sara"}, "booking_id": 7})
    assert "already booked" in prompt and PHONE in prompt


def test_no_booking_guidance_outside_a_booking(monkeypatch):
    _, prompt = _reply(monkeypatch, {"intent": "FAQ"})
    assert "Booking in progress" not in prompt


def test_reply_knows_the_booking_was_saved_on_the_same_turn(fake_llms, saved_bookings, agent):
    fakes = fake_llms(turns=[understood("BOOK", name="Sara", date="Friday", time="10am", phone="0300")],
                      replies=["ok"])
    agent("book Sara for Friday 10am, phone 0300")
    # understanding + save run before the reply, so the reply prompt already knows
    assert "has just been saved" in fakes["responder"].calls[0][0].content


# ── Spoken output ──────────────────────────────────────────────

def test_markdown_is_stripped_before_text_to_speech(monkeypatch):
    reply, _ = _reply(monkeypatch, {"intent": "FAQ"}, "Here you go:\n- **Name:** Sara\n- **Time:** 10am\n1. Arrive early")
    assert reply == "Here you go: Name: Sara Time: 10am Arrive early"


def test_plain_replies_are_unchanged(monkeypatch):
    reply, _ = _reply(monkeypatch, {"intent": "FAQ"}, "We open at 9 AM - 7 PM, Monday to Saturday.")
    assert reply == "We open at 9 AM - 7 PM, Monday to Saturday."


# ── Retrieval relevance cut-off ────────────────────────────────

def test_retriever_drops_chunks_beyond_the_distance_cutoff(monkeypatch):
    class FakeCollection:
        def query(self, query_texts, n_results):
            return {"documents": [["close", "borderline", "far"]], "distances": [[0.5, 0.88, 0.95]]}

    monkeypatch.setattr(retriever, "get_collection", lambda: FakeCollection())
    assert retriever.retrieve("anything") == "close\n\nborderline"
