"""Booking flow through the real graph. The clock is frozen at Thursday 2026-11-26 14:00 (conftest)."""
import app.integrations.booking_store as booking_store
from tests.fakes import understood

FRIDAY, SATURDAY, SUNDAY, MONDAY, TUESDAY = "2026-11-27", "2026-11-28", "2026-11-29", "2026-11-30", "2026-12-01"
PHONE = "03001234567"


def test_booking_is_saved_and_emailed_exactly_once(fake_llms, saved_bookings, agent):
    fake_llms(
        turns=[
            understood("BOOK", service="cleaning"),
            understood("COLLECT", name="Sara Khan"),
            understood("COLLECT", date=FRIDAY, time="10:00"),
            understood("FAQ"),
            understood("COLLECT", phone="0300 123 4567"),
            understood("COLLECT"),
        ],
        replies=[f"reply {i}" for i in range(6)],
    )

    results = [agent(text) for text in [
        "I want to book a cleaning", "Sara Khan", "Friday at 10am",
        "do you take insurance?", "0300 123 4567", "yes thanks",
    ]]

    assert [r.get("new_booking_id") for r in results] == [None, None, None, None, 1, None]
    assert len(saved_bookings["bookings"]) == 1
    assert len(saved_bookings["emails"]) == 1
    assert saved_bookings["bookings"][0] == {
        "service": "cleaning", "name": "Sara Khan", "date": FRIDAY, "time": "10:00",
        "phone": PHONE, "complete": True,  # phone normalised to digits
    }


def test_details_from_several_turns_are_merged(fake_llms, saved_bookings, agent):
    fake_llms(turns=[understood("BOOK", name="Sara Khan"), understood("COLLECT", date=MONDAY)], replies=["a", "b"])
    agent("book me in, I'm Sara Khan")
    assert agent("Monday")["booking_data"] == {"name": "Sara Khan", "date": MONDAY, "complete": False}


def test_a_correction_replaces_the_earlier_value(fake_llms, saved_bookings, agent):
    fake_llms(turns=[understood("BOOK", date=FRIDAY), understood("COLLECT", date=SATURDAY)], replies=["a", "b"])
    agent("book Friday")
    assert agent("actually make it Saturday")["booking_data"]["date"] == SATURDAY


def test_booking_details_are_ignored_on_non_booking_turns(fake_llms, saved_bookings, agent):
    fake_llms(turns=[understood("BOOK", name="Sara Khan"), understood("FAQ", date=MONDAY)], replies=["a", "b"])
    agent("book me in, I'm Sara Khan")
    result = agent("are you open on Monday?")  # a question, not a booking detail
    assert result["booking_data"] == {"name": "Sara Khan", "complete": False}


def test_a_saved_booking_is_never_changed(fake_llms, saved_bookings, agent):
    fake_llms(
        turns=[understood("BOOK", name="A", date=MONDAY, time="09:00", phone=PHONE),
               understood("COLLECT", date=TUESDAY)],
        replies=["a", "b"],
    )
    agent("book A for Monday 9am")
    result = agent("can you move it to Tuesday?")
    assert result["booking_data"]["date"] == MONDAY
    assert len(saved_bookings["bookings"]) == 1


def test_completeness_needs_name_date_time_and_phone(fake_llms, saved_bookings, agent):
    fake_llms(turns=[understood("BOOK", name="Sara", date=MONDAY, time="09:00", email="s@example.com")], replies=["a"])
    result = agent("book Sara Monday 9am, email s@example.com")
    assert result["booking_data"]["complete"] is False  # phone still missing
    assert saved_bookings["bookings"] == []


def test_blank_and_null_strings_are_not_treated_as_details(fake_llms, saved_bookings, agent):
    fake_llms(turns=[understood("BOOK", name="  ", email="null", service="cleaning")], replies=["a"])
    assert agent("book a cleaning")["booking_data"] == {"service": "cleaning", "complete": False}


def test_separate_sessions_do_not_share_bookings(fake_llms, saved_bookings, agent):
    fake_llms(turns=[understood("BOOK", name="A", date=MONDAY, time="09:00", phone=PHONE),
                     understood("BOOK", name="B", date=MONDAY, time="11:00", phone=PHONE)], replies=["a", "b"])
    agent("book", session_id="caller-1")
    agent("book", session_id="caller-2")
    assert len(saved_bookings["bookings"]) == 2


def test_one_llm_call_to_understand_each_turn(fake_llms, saved_bookings, agent):
    fakes = fake_llms(turns=[understood("BOOK", name="Sara")], replies=["a"])
    agent("book me in, I'm Sara")
    assert len(fakes["understand"].calls) == 1 and len(fakes["responder"].calls) == 1


def test_understand_is_told_todays_date(fake_llms, saved_bookings, agent):
    fakes = fake_llms(turns=[understood("BOOK")], replies=["a"])
    agent("book me in for tomorrow")
    assert "Today is Thursday 2026-11-26, 14:00" in fakes["understand"].calls[0][1].content


# ── rejected details: dropped, with a reason the reply passes on ──

def _reject(fake_llms, agent, previous, given):
    fake_llms(turns=[understood("BOOK", **previous), understood("COLLECT", **given)], replies=["a", "b"])
    agent("first")
    return agent("second")


def test_closed_day_is_rejected(fake_llms, saved_bookings, agent):
    result = _reject(fake_llms, agent, {"name": "Sara"}, {"date": SUNDAY})
    assert result["booking_issue"] == "We're closed on Sundays."
    assert "date" not in result["booking_data"]


def test_past_date_is_rejected(fake_llms, saved_bookings, agent):
    result = _reject(fake_llms, agent, {"name": "Sara"}, {"date": "2026-11-20"})
    assert "already passed" in result["booking_issue"]


def test_time_outside_opening_hours_drops_only_the_time(fake_llms, saved_bookings, agent):
    result = _reject(fake_llms, agent, {"date": FRIDAY}, {"time": "18:45"})  # a 30-minute slot would end after 7 PM
    assert "from 9 AM until 7 PM" in result["booking_issue"]
    assert result["booking_data"]["date"] == FRIDAY and "time" not in result["booking_data"]


def test_a_taken_slot_is_rejected(fake_llms, saved_bookings, agent):
    booking_store.save_booking("someone-else", "City Dental Clinic",
                               {"name": "X", "phone": PHONE, "date": FRIDAY, "time": "10:00"})
    result = _reject(fake_llms, agent, {"date": FRIDAY}, {"time": "10:15"})  # overlaps the 10:00 slot
    assert result["booking_issue"] == "That time is already booked."


def test_unreadable_phone_is_rejected(fake_llms, saved_bookings, agent):
    result = _reject(fake_llms, agent, {"name": "Sara"}, {"phone": "no thanks"})
    assert "phone number" in result["booking_issue"] and "phone" not in result["booking_data"]


def test_unparseable_date_is_rejected(fake_llms, saved_bookings, agent):
    result = _reject(fake_llms, agent, {"name": "Sara"}, {"date": "sometime soon"})
    assert result["booking_issue"] == "I didn't catch the date."


def test_spoken_dates_are_resolved_in_code(fake_llms, saved_bookings, agent):
    fake_llms(turns=[understood("BOOK", date="next Monday", time="15:30")], replies=["a"])
    result = agent("next Monday at half three")
    assert result["booking_data"]["date"] == MONDAY and result["booking_data"]["time"] == "15:30"


def test_issue_is_cleared_on_the_next_turn(fake_llms, saved_bookings, agent):
    fake_llms(turns=[understood("BOOK", date=SUNDAY), understood("COLLECT", date=MONDAY)], replies=["a", "b"])
    assert agent("Sunday")["booking_issue"]
    assert agent("Monday then")["booking_issue"] is None


def test_slot_taken_at_the_moment_of_saving(fake_llms, agent, monkeypatch):
    """Two callers confirm the same slot together: the database check at save time catches it."""
    import app.agent.nodes.booking_saver as saver_node

    emails = []
    monkeypatch.setattr(saver_node, "send_booking_email", lambda *args: emails.append(args))
    monkeypatch.setattr(saver_node, "save_booking", lambda *args: None)  # lost the race
    fake_llms(turns=[understood("BOOK", name="A", date=FRIDAY, time="10:00", phone=PHONE)], replies=["a"])
    result = agent("book A Friday 10am")
    assert result.get("new_booking_id") is None and emails == []
    assert result["booking_issue"] == "Sorry, someone just booked that time."
    assert "time" not in result["booking_data"] and result["booking_data"]["complete"] is False
