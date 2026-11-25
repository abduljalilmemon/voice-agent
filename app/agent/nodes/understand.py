"""One LLM call per turn: classify the latest message AND extract any booking details from it."""
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.agent.state import AgentState
from app.config import BIZ_CONFIG
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
Keep dates and times as the customer said them. If they correct an earlier detail, give the new value.

Examples (agent's last message | latest user message => result):
(start) | "hi there" => GREETING
(start) | "what are your hours?" => FAQ
(start) | "i want to book a cleaning" => BOOK, service=cleaning
(start) | "book me in for Friday at 10, I'm Sara Khan" => BOOK, name=Sara Khan, date=Friday, time=10
"Could I get your full name?" | "Abdul Jalil" => COLLECT, name=Abdul Jalil
"What date and time works for you?" | "tomorrow at 3pm" => COLLECT, date=tomorrow, time=3pm
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

def _format_input(state: AgentState) -> str:
    *previous, latest = state["messages"]
    history = [f"{_speaker(m)}: {m.content}" for m in previous[-HISTORY_WINDOW:]]
    parts = ["Recent conversation:\n" + ("\n".join(history) if history else "(start of conversation)")]
    if status := _booking_status(state):
        parts.append(status)
    parts.append(f"Latest user message: {latest.content}")
    return "\n\n".join(parts)

def understand(state: AgentState) -> AgentState:
    result: TurnUnderstanding = llm.invoke([
        SystemMessage(content=UNDERSTAND_PROMPT),
        HumanMessage(content=_format_input(state)),
    ])
    update: AgentState = {"intent": result.intent}
    logger.info("Intent: %s", result.intent)

    # booking details only count on booking turns, and never change a booking that's already saved
    if result.intent in {"BOOK", "COLLECT"} and not state.get("booking_id"):
        new_details = {k: v.strip() for k, v in result.booking.model_dump().items()
                       if isinstance(v, str) and v.strip() and v.strip().lower() != "null"}
        data = {**(state.get("booking_data") or {}), **new_details}
        data["complete"] = all(data.get(field) for field in REQUIRED_FIELDS)
        update["booking_data"] = data
        logger.info("Booking: complete=%s", data["complete"])
        logger.debug("Booking data: %s", data)
    return update
