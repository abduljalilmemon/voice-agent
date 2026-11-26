"""One LLM call per turn: classify the latest message AND extract any booking details from it."""
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.agent.state import AgentState
from app.booking_rules import (
    day_problem,
    normalize_phone,
    now_local,
    parse_date,
    parse_time,
    resolve_date,
    slot_problem,
    valid_email,
)
from app.config import BIZ_CONFIG
from app.integrations.booking_store import is_slot_taken
from app.llm import get_llm

logger = logging.getLogger(__name__)

Intent = Literal["GREETING", "FAQ", "BOOK", "COLLECT", "COMPLAINT", "ESCALATE", "CONVERSATION", "OUT_OF_SCOPE"]

# The schema is sent with every request, so it stays lean: field meanings are in the prompt, and
# there are no docstrings (they'd be sent too). Every field is required but nullable, which suits
# strict JSON-schema structured output.
class BookingDetails(BaseModel):
    name: str | None
    phone: str | None
    email: str | None
    date: str | None
    time: str | None
    service: str | None

class TurnUnderstanding(BaseModel):
    intent: Intent
    booking: BookingDetails

# deterministic, and little reasoning needed to label one message -- cheaper and faster
llm = get_llm(temperature=0, schema=TurnUnderstanding, reasoning_effort="low")

# how many previous messages are shown (2 exchanges is enough to read short answers)
HISTORY_WINDOW = 4

# a booking is complete when these are known; completeness is decided here, never by the LLM
REQUIRED_FIELDS = ("name", "date", "time", "phone")

UNDERSTAND_PROMPT = f"""You analyse the LATEST user message for the voice assistant of {BIZ_CONFIG["business_name"]},
a {BIZ_CONFIG["business_type"]} (appointments, services, prices, hours, location).

1. intent -- classify only the latest message, using the conversation to interpret short answers:
GREETING     = hello, introductions, thanks, small talk, closing the chat ("no, that's all")
FAQ          = a question about the business: services, prices, hours, location, insurance, staff
BOOK         = starts a new booking request
COLLECT      = answers the agent's booking question: name, phone, email, date, time, service, yes/no confirmation
COMPLAINT    = frustration or a problem with the business
ESCALATE     = asks for a human
CONVERSATION = asks about something said earlier in this chat
OUT_OF_SCOPE = anything unrelated to the business

A short answer is COLLECT only if the agent just asked for booking details, or a booking is in
progress and the answer gives a booking detail. A business question asked in the middle of a
booking is still FAQ.

2. booking -- booking details given in the LATEST message only; null for anything not in it.
date: the day exactly as the customer said it ("this Friday", "tomorrow", "the 2nd of December"); do not convert it.
time: 24-hour HH:MM ("10am" = 10:00, "3:30pm" = 15:30). If they correct an earlier detail, give the new value.

Examples (agent's last message | latest user message => result):
(start) | "hi there" => GREETING
(start) | "what are your hours?" => FAQ
(start) | "i want to book a cleaning" => BOOK, service=cleaning
(start) | "book me in for Friday at 10, I'm Sara Khan" => BOOK, name=Sara Khan, date=Friday, time=10:00
"Could I get your full name?" | "Abdul Jalil" => COLLECT, name=Abdul Jalil
"What date and time works for you?" | "tomorrow at 3pm" => COLLECT, date=tomorrow, time=15:00
"Could I get your phone number?" | "03477733113" => COLLECT, phone=03477733113
"Shall I go ahead and book that?" | "yes" => COLLECT
"Is there anything else I can help you with?" | "no that's it, thanks" => GREETING
"Could I get your phone number?" | "how much is a cleaning?" => FAQ
"Could I get your full name?" | "what's the capital of france?" => OUT_OF_SCOPE
(start) | "this is frustrating, nobody called me back" => COMPLAINT
(start) | "let me talk to a human" => ESCALATE
"Nice to meet you, Ali! How can I help?" | "what is my name?" => CONVERSATION"""

def _speaker(message) -> str:
    return "User" if isinstance(message, HumanMessage) else "Agent"

def _booking_status(state: AgentState) -> str:
    data = state.get("booking_data") or {}
    if state.get("booking_id") or not any(data.get(f) for f in REQUIRED_FIELDS):
        return ""
    known = ", ".join(f"{f}={data[f]}" for f in ("name", "service", "date", "time", "phone", "email") if data.get(f))
    return f"A booking is in progress. Collected so far: {known}."

def _format_input(state: AgentState, now) -> str:
    *previous, latest = state["messages"]
    history = [f"{_speaker(m)}: {m.content}" for m in previous[-HISTORY_WINDOW:]]
    parts = [f"Today is {now:%A %Y-%m-%d}, {now:%H:%M} at the clinic.",
             "Recent conversation:\n" + ("\n".join(history) if history else "(start of conversation)")]
    if status := _booking_status(state):
        parts.append(status)
    parts.append(f"Latest user message: {latest.content}")
    return "\n\n".join(parts)

def _apply_details(previous: dict, given: dict, now) -> tuple[dict, str | None]:
    """Merge newly given details into the booking, keeping only valid, bookable ones.

    Returns the booking and, if something was rejected, a short reason the reply can pass on.
    """
    data, issue = dict(previous), None
    for field, value in given.items():
        if field == "phone":
            if not (value := normalize_phone(value)):
                issue = "That doesn't sound like a complete phone number."
                continue
        elif field == "email" and not valid_email(value):
            issue = "That email address doesn't look right."
            continue
        elif field == "date":
            if not (day := resolve_date(value, now)):  # "this Friday" -> a real date, in code
                issue = "I didn't catch the date."
                continue
            value = day.isoformat()
        elif field == "time":
            if not (at := parse_time(value)):
                issue = "I didn't catch the time."
                continue
            value = at.strftime("%H:%M")
        data[field] = value

    # check the day / slot as soon as enough of it is known
    day, at = parse_date(data.get("date")), parse_time(data.get("time"))
    problem = None
    if day and at:
        problem = slot_problem(day, at, now)
        if not problem and is_slot_taken(day.isoformat(), at.strftime("%H:%M")):
            problem = "That time is already booked."
    elif day:
        problem = day_problem(day, now)
    if problem:
        issue = problem
        just_given = {"date", "time"} & given.keys()
        for field in just_given or {"date", "time"}:  # nothing new? the old slot became invalid
            data.pop(field, None)

    data["complete"] = all(data.get(field) for field in REQUIRED_FIELDS)
    return data, issue

def understand(state: AgentState) -> AgentState:
    now = now_local()
    result: TurnUnderstanding = llm.invoke([
        SystemMessage(content=UNDERSTAND_PROMPT),
        HumanMessage(content=_format_input(state, now)),
    ])
    update: AgentState = {"intent": result.intent}
    logger.info("Intent: %s", result.intent)

    # booking details only count on booking turns, and never change a booking that's already saved
    if result.intent in {"BOOK", "COLLECT"} and not state.get("booking_id"):
        given = {k: v.strip() for k, v in result.booking.model_dump().items()
                 if isinstance(v, str) and v.strip() and v.strip().lower() != "null"}
        data, issue = _apply_details(state.get("booking_data") or {}, given, now)
        update["booking_data"] = data
        update["booking_issue"] = issue
        logger.info("Booking: complete=%s%s", data["complete"], f" (rejected: {issue})" if issue else "")
        logger.debug("Booking data: %s", data)
    return update
