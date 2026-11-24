import json

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes.booking_extractor import extract_booking


def booking(name=None, date=None, time=None, phone=None, complete=False):
    return json.dumps({"name": name, "phone": phone, "email": None, "date": date,
                       "time": time, "service": None, "complete": complete})


def test_booking_is_saved_and_emailed_exactly_once(fake_llms, saved_bookings, agent):
    fake_llms(
        intents=["BOOK", "COLLECT", "COLLECT", "FAQ", "COLLECT", "COLLECT"],
        replies=[f"reply {i}" for i in range(6)],
        # the extractor must not run on the FAQ turn or after the booking is saved
        extractions=[
            booking(),
            booking(name="Sara Khan"),
            booking(name="Sara Khan", date="Friday", time="10am"),
            booking(name="Sara Khan", date="Friday", time="10am", phone="03001234567"),
        ],
    )

    results = [agent(text) for text in [
        "I want to book a cleaning", "Sara Khan", "Friday at 10am",
        "do you take insurance?", "03001234567", "yes thanks",
    ]]

    assert [r.get("new_booking_id") for r in results] == [None, None, None, None, 1, None]
    assert len(saved_bookings["bookings"]) == 1
    assert len(saved_bookings["emails"]) == 1
    assert saved_bookings["bookings"][0]["phone"] == "03001234567"


def test_booking_data_survives_a_non_booking_turn(fake_llms, saved_bookings, agent):
    fake_llms(intents=["BOOK", "FAQ"], replies=["a", "b"], extractions=[booking(name="Sara Khan")])
    agent("book me in, I'm Sara Khan")
    result = agent("what are your hours?")
    assert result["booking_data"]["name"] == "Sara Khan"


def test_separate_sessions_do_not_share_bookings(fake_llms, saved_bookings, agent):
    complete = booking(name="A", date="Mon", time="9am", phone="1")
    fake_llms(intents=["BOOK", "BOOK"], replies=["a", "b"], extractions=[complete, complete])
    agent("book", session_id="caller-1")
    agent("book", session_id="caller-2")
    assert len(saved_bookings["bookings"]) == 2


def _extract(llm_output, monkeypatch):
    import app.agent.nodes.booking_extractor as extractor_node
    from tests.fakes import FakeLLM

    monkeypatch.setattr(extractor_node, "llm", FakeLLM([llm_output]))
    state = {"intent": "COLLECT", "messages": [HumanMessage(content="hi"), AIMessage(content="hello")]}
    return extract_booking(state)


def test_completeness_is_computed_in_code_not_trusted_from_the_llm(monkeypatch):
    # the LLM claims complete=true, but the phone number is missing
    result = _extract(booking(name="Sara", date="Mon", time="9am", complete=True), monkeypatch)
    assert result["booking_data"]["complete"] is False


def test_extractor_handles_markdown_fences_and_null_strings(monkeypatch):
    raw = "```json\n" + json.dumps({"name": "Sara", "phone": "null", "date": "Mon", "time": ""}) + "\n```"
    data = _extract(raw, monkeypatch)["booking_data"]
    assert data["name"] == "Sara"
    assert data["phone"] is None and data["time"] is None
    assert data["complete"] is False


def test_unparseable_extraction_keeps_previously_collected_data(monkeypatch):
    # returning {} leaves booking_data in the checkpoint untouched
    assert _extract("sorry, I can't do that", monkeypatch) == {}
