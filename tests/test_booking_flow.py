from tests.fakes import understood


def test_booking_is_saved_and_emailed_exactly_once(fake_llms, saved_bookings, agent):
    fake_llms(
        turns=[
            understood("BOOK", service="cleaning"),
            understood("COLLECT", name="Sara Khan"),
            understood("COLLECT", date="Friday", time="10am"),
            understood("FAQ"),
            understood("COLLECT", phone="03001234567"),
            understood("COLLECT"),
        ],
        replies=[f"reply {i}" for i in range(6)],
    )

    results = [agent(text) for text in [
        "I want to book a cleaning", "Sara Khan", "Friday at 10am",
        "do you take insurance?", "03001234567", "yes thanks",
    ]]

    assert [r.get("new_booking_id") for r in results] == [None, None, None, None, 1, None]
    assert len(saved_bookings["bookings"]) == 1
    assert len(saved_bookings["emails"]) == 1
    assert saved_bookings["bookings"][0] == {
        "service": "cleaning", "name": "Sara Khan", "date": "Friday", "time": "10am",
        "phone": "03001234567", "complete": True,
    }


def test_details_from_several_turns_are_merged(fake_llms, saved_bookings, agent):
    fake_llms(turns=[understood("BOOK", name="Sara Khan"), understood("COLLECT", date="Monday")],
              replies=["a", "b"])
    agent("book me in, I'm Sara Khan")
    result = agent("Monday")
    assert result["booking_data"] == {"name": "Sara Khan", "date": "Monday", "complete": False}


def test_a_correction_replaces_the_earlier_value(fake_llms, saved_bookings, agent):
    fake_llms(turns=[understood("BOOK", date="Friday"), understood("COLLECT", date="Saturday")],
              replies=["a", "b"])
    agent("book Friday")
    assert agent("actually make it Saturday")["booking_data"]["date"] == "Saturday"


def test_booking_details_are_ignored_on_non_booking_turns(fake_llms, saved_bookings, agent):
    fake_llms(turns=[understood("BOOK", name="Sara Khan"), understood("FAQ", date="Sunday")],
              replies=["a", "b"])
    agent("book me in, I'm Sara Khan")
    result = agent("are you open on Sunday?")  # a question, not a booking detail
    assert result["booking_data"] == {"name": "Sara Khan", "complete": False}


def test_a_saved_booking_is_never_changed(fake_llms, saved_bookings, agent):
    fake_llms(
        turns=[understood("BOOK", name="A", date="Mon", time="9am", phone="1"), understood("COLLECT", date="Tue")],
        replies=["a", "b"],
    )
    agent("book A for Monday 9am, phone 1")
    result = agent("can you move it to Tuesday?")
    assert result["booking_data"]["date"] == "Mon"
    assert len(saved_bookings["bookings"]) == 1


def test_completeness_needs_name_date_time_and_phone(fake_llms, saved_bookings, agent):
    fake_llms(turns=[understood("BOOK", name="Sara", date="Mon", time="9am", email="s@example.com")],
              replies=["a"])
    result = agent("book Sara Monday 9am, email s@example.com")
    assert result["booking_data"]["complete"] is False  # phone still missing
    assert saved_bookings["bookings"] == []


def test_blank_and_null_strings_are_not_treated_as_details(fake_llms, saved_bookings, agent):
    fake_llms(turns=[understood("BOOK", name="  ", phone="null", service="cleaning")], replies=["a"])
    assert agent("book a cleaning")["booking_data"] == {"service": "cleaning", "complete": False}


def test_separate_sessions_do_not_share_bookings(fake_llms, saved_bookings, agent):
    complete = dict(name="A", date="Mon", time="9am", phone="1")
    fake_llms(turns=[understood("BOOK", **complete), understood("BOOK", **complete)], replies=["a", "b"])
    agent("book", session_id="caller-1")
    agent("book", session_id="caller-2")
    assert len(saved_bookings["bookings"]) == 2


def test_one_llm_call_to_understand_each_turn(fake_llms, saved_bookings, agent):
    fakes = fake_llms(turns=[understood("BOOK", name="Sara")], replies=["a"])
    agent("book me in, I'm Sara")
    assert len(fakes["understand"].calls) == 1 and len(fakes["responder"].calls) == 1
