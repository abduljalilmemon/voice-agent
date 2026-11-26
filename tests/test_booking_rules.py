from datetime import date, datetime, time

import pytest
from langchain_core.messages import HumanMessage

import app.agent.nodes.responder as responder
import app.integrations.booking_store as booking_store
from app import booking_rules as rules
from tests.fakes import FakeLLM

NOW = datetime(2026, 11, 26, 14, 0)  # Thursday


@pytest.mark.parametrize("day, at, problem", [
    (date(2026, 11, 27), time(10, 0), None),                      # Friday morning
    (date(2026, 11, 28), time(18, 30), None),                     # last slot on Saturday
    (date(2026, 11, 29), time(10, 0), "We're closed on Sundays."),
    (date(2026, 11, 20), time(10, 0), "Friday 20 November has already passed."),
    (date(2026, 11, 26), time(13, 0), "That time has already passed."),   # earlier today
    (date(2026, 11, 27), time(18, 45), "On Fridays we take appointments from 9 AM until 7 PM."),
    (date(2026, 11, 27), time(8, 30), "On Fridays we take appointments from 9 AM until 7 PM."),
])
def test_slot_problem(day, at, problem):
    assert rules.slot_problem(day, at, NOW) == problem


@pytest.mark.parametrize("raw, expected", [
    ("0300 123 4567", "03001234567"),
    ("+92 300 1234567", "+923001234567"),
    ("(021) 123-4567", "0211234567"),
    ("no thanks", None),
    ("123", None),
    ("1" * 16, None),
])
def test_normalize_phone(raw, expected):
    assert rules.normalize_phone(raw) == expected


def test_parsing_accepts_only_iso():
    assert rules.parse_date("2026-11-27") == date(2026, 11, 27)
    assert rules.parse_date("Friday") is None
    assert rules.parse_time("15:30") == time(15, 30)
    assert rules.parse_time("3:30pm") is None


@pytest.mark.parametrize("said, expected", [
    ("2026-11-27", date(2026, 11, 27)),
    ("this Friday", date(2026, 11, 27)),
    ("Friday", date(2026, 11, 27)),
    ("next Monday", date(2026, 11, 30)),            # an LLM resolved this to a Friday
    ("tomorrow", date(2026, 11, 27)),
    ("day after tomorrow", date(2026, 11, 28)),
    ("the 2nd of December", date(2026, 12, 2)),
    ("December 2", date(2026, 12, 2)),
    ("2/12", date(2026, 12, 2)),                     # day first, as in Pakistan
    ("the 14th", date(2026, 12, 14)),                # 14 November has passed: the next 14th
    ("the 30th", date(2026, 11, 30)),
    ("sometime soon", None),
])
def test_resolve_spoken_dates(said, expected):
    assert rules.resolve_date(said, NOW) == expected


def test_readable_formats_and_hours_text():
    assert rules.format_date(date(2026, 11, 27)) == "Friday 27 November"
    assert rules.format_time(time(10, 0)) == "10 AM" and rules.format_time(time(15, 30)) == "3:30 PM"
    assert rules.opening_hours_text() == "Monday to Saturday, 9 AM to 7 PM. Closed on Sunday."
    assert rules.valid_email("a@b.co") and not rules.valid_email("not an email")


def test_overlapping_slots_are_taken():
    booking_store.save_booking("s1", "City Dental Clinic", {"name": "A", "date": "2026-11-27", "time": "10:00"})
    assert booking_store.is_slot_taken("2026-11-27", "10:00")
    assert booking_store.is_slot_taken("2026-11-27", "10:29")
    assert not booking_store.is_slot_taken("2026-11-27", "10:30")    # starts as the other ends
    assert not booking_store.is_slot_taken("2026-11-28", "10:00")    # another day


def test_save_refuses_a_taken_slot():
    first = booking_store.save_booking("s1", "Clinic", {"name": "A", "date": "2026-11-27", "time": "10:00"})
    second = booking_store.save_booking("s2", "Clinic", {"name": "B", "date": "2026-11-27", "time": "10:15"})
    assert first is not None and second is None


def test_legacy_free_text_bookings_do_not_break_slot_checks():
    booking_store.save_booking("old", "Clinic", {"name": "A", "date": "2026-11-27", "time": "10am"})
    assert not booking_store.is_slot_taken("2026-11-27", "10:00")


def _prompt(monkeypatch, state):
    fake = FakeLLM(["ok"])
    monkeypatch.setattr(responder, "llm", fake)
    responder.generate_response({"messages": [HumanMessage(content="hi")], **state})
    return fake.calls[0][0].content


def test_reply_prompt_says_dates_naturally(monkeypatch):
    prompt = _prompt(monkeypatch, {"intent": "COLLECT", "booking_data": {"name": "Sara", "date": "2026-11-27", "time": "10:00"}})
    assert "date: Friday 27 November, time: 10 AM" in prompt


def test_reply_prompt_passes_on_why_a_detail_was_rejected(monkeypatch):
    prompt = _prompt(monkeypatch, {"intent": "COLLECT", "booking_data": {"name": "Sara"},
                                   "booking_issue": "We're closed on Sundays."})
    assert "couldn't be used: We're closed on Sundays." in prompt
    assert "ask for an alternative" in prompt
