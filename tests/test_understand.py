from langchain_core.messages import AIMessage, HumanMessage

import app.agent.nodes.understand as understand_node
from tests.fakes import FakeLLM, understood


def _understand(monkeypatch, messages, result=None, **state):
    fake = FakeLLM([result or understood("COLLECT")])
    monkeypatch.setattr(understand_node, "llm", fake)
    update = understand_node.understand({"messages": messages, **state})
    system, user = fake.calls[0]
    return update, system.content, user.content


def test_prompt_has_recent_history_and_examples(monkeypatch):
    history = [HumanMessage(content=f"user {i}") if i % 2 == 0 else AIMessage(content=f"agent {i}")
               for i in range(6)]
    update, system, user = _understand(monkeypatch, history + [HumanMessage(content="Sara Khan")])

    assert update["intent"] == "COLLECT"
    assert "Examples" in system and "City Dental Clinic" in system
    assert "Latest user message: Sara Khan" in user
    assert "Agent: agent 5" in user and "User: user 2" in user  # last 4 messages
    assert "user 0" not in user and "agent 1" not in user   # older ones dropped


def test_first_message_has_no_history(monkeypatch):
    _, _, user = _understand(monkeypatch, [HumanMessage(content="hi")], understood("GREETING"))
    assert "(start of conversation)" in user


def test_told_when_a_booking_is_in_progress(monkeypatch):
    _, _, user = _understand(monkeypatch, [HumanMessage(content="0300 1234567")],
                             booking_data={"name": "Sara", "date": "Friday"})
    assert "A booking is in progress. Collected so far: name=Sara, date=Friday." in user


def test_not_told_once_the_booking_is_saved(monkeypatch):
    _, _, user = _understand(monkeypatch, [HumanMessage(content="thanks")],
                             booking_data={"name": "Sara"}, booking_id=3)
    assert "booking is in progress" not in user


def test_non_booking_turns_leave_booking_data_untouched(monkeypatch):
    update, _, _ = _understand(monkeypatch, [HumanMessage(content="hi")], understood("GREETING", name="Sara"))
    assert update == {"intent": "GREETING"}


def test_schema_is_strict_friendly():
    # strict JSON-schema structured output wants every property required (nullable, not optional)
    schema = understand_node.BookingDetails.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"])
