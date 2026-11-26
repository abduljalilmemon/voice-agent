import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from app.agent.state import AgentState
from app.booking_rules import format_date, format_time, now_local, opening_hours_text, parse_date, parse_time
from app.config import BIZ_CONFIG
from app.llm import get_llm

logger = logging.getLogger(__name__)

llm = get_llm()

NAME = BIZ_CONFIG["business_name"]
PHONE = BIZ_CONFIG["phone"]
HOURS = opening_hours_text()

BASE_PROMPT = f"""You are the voice assistant for {NAME}, a {BIZ_CONFIG["business_type"]}. You answer customer calls and chats.

Business details:
- Address: {BIZ_CONFIG["address"]}
- Phone: {PHONE}
- Email: {BIZ_CONFIG["email"]}
- Hours: {HOURS} Appointments last {BIZ_CONFIG["appointment_minutes"]} minutes.

Your replies are spoken aloud by text-to-speech:
- 1 to 3 short sentences of plain, natural conversation.
- No markdown, bullet points, numbered lists or emoji.

Only help with things related to {NAME}. Never invent prices, services, staff or policies: if the
information you have doesn't answer the question, say so and suggest calling {PHONE}.
You may use anything the customer told you earlier in this conversation, such as their name."""

INTENT_GUIDANCE = {
    "GREETING": f"Respond warmly to the greeting, thanks or goodbye. If greeting, introduce yourself as {NAME}'s assistant.",
    "FAQ": "Answer the question using the business details and any relevant information below.",
    "COMPLAINT": f"The customer is unhappy. Apologise sincerely, acknowledge the problem, and offer that the team can sort it out on {PHONE}.",
    "CONVERSATION": "The customer is asking about this conversation. Answer from the conversation history.",
    "BOOK": "The customer wants an appointment.",
    "COLLECT": "The customer is giving details for their appointment.",
}

STATIC_RESPONSES = {
    "OUT_OF_SCOPE": f"I can only help with questions about {NAME}, like our services, prices, opening hours or appointments. What can I help you with?",
    # no live transfer exists, so be honest and give a way to reach a person
    "ESCALATE": f"I can't transfer you from here, but you can reach our team directly on {PHONE}. {HOURS} Is there anything I can help with in the meantime?",
}

# asked in this order, one at a time
BOOKING_FIELDS = [("name", "their full name"), ("date", "the preferred date"),
                  ("time", "the preferred time"), ("phone", "a phone number")]

def _readable(field: str, value: str) -> str:
    """Stored as ISO (unambiguous); said as "Friday 27 November" / "10 AM"."""
    if field == "date" and (d := parse_date(value)):
        return format_date(d)
    if field == "time" and (t := parse_time(value)):
        return format_time(t)
    return value

def booking_guidance(state: AgentState) -> str:
    """Tell the model the real booking state, so it never has to guess whether it's booked."""
    data = state.get("booking_data") or {}
    summary = ", ".join(f"{field}: {_readable(field, data[field])}"
                        for field in ("name", "service", "date", "time", "phone") if data.get(field))
    issue = state.get("booking_issue")
    problem = (f"The customer's last request couldn't be used: {issue} Tell them briefly and kindly, "
               f"then ask for an alternative. " if issue else "")

    if state.get("new_booking_id"):
        return (f"The appointment request has just been saved ({summary}). Confirm the details back to "
                f"the customer and say the clinic will call them to confirm.")
    if state.get("booking_id"):
        return (f"An appointment was already booked in this conversation ({summary}). Changes can't be made "
                f"here, so for changes ask them to call {PHONE}.")

    if not data and state.get("intent") not in {"BOOK", "COLLECT"}:
        return ""  # no booking in progress
    missing = [label for field, label in BOOKING_FIELDS if not data.get(field)]
    if "the preferred date" in missing and "the preferred time" in missing:
        missing.remove("the preferred time")
        missing[missing.index("the preferred date")] = "the preferred date and time"
    if not missing:
        return problem
    collected = f"Collected so far: {summary}. " if summary else ""
    return (f"Booking in progress. {problem}{collected}Ask only for the next missing item: {missing[0]}. "
            f"The appointment is NOT booked yet, so never say it is confirmed or booked.")

def spoken(text: str) -> str:
    """Safety net for text-to-speech: strip any markdown the model produced anyway."""
    text = re.sub(r"\*\*|__|`|^#+\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*(?:[-*•]|\d+\.)\s+", "", text, flags=re.M)
    return re.sub(r"\s*\n+\s*", " ", text).strip()

# The reply sees only recent messages (booking facts travel separately in booking_data), and the
# stored conversation is capped, so long chats don't grow slower, costlier and unbounded on disk.
REPLY_HISTORY = 12
MAX_STORED_MESSAGES = 40

def recent_messages(messages: list, limit: int = REPLY_HISTORY) -> list:
    """The last `limit` messages, starting with a customer message (some providers require that)."""
    recent = messages[-limit:]
    while recent and not isinstance(recent[0], HumanMessage):
        recent = recent[1:]
    return recent

def _with_pruning(state: AgentState, reply: AIMessage) -> AgentState:
    """Add the reply, removing the oldest stored messages beyond the cap."""
    messages = state["messages"]
    excess = len(messages) + 1 - MAX_STORED_MESSAGES
    removals = [RemoveMessage(id=m.id) for m in messages[:max(0, excess)] if m.id]
    return {"messages": [*removals, reply]}

def generate_response(state: AgentState) -> AgentState:
    intent = state.get("intent", "FAQ")

    # return static response immediately -- no LLM call
    if intent in STATIC_RESPONSES:
        logger.debug("Static reply for %s", intent)
        return _with_pruning(state, AIMessage(content=STATIC_RESPONSES[intent]))

    now = now_local()
    parts = [BASE_PROMPT, f"Today is {format_date(now.date())} {now.year}, {format_time(now.time().replace(second=0, microsecond=0))}.",
             INTENT_GUIDANCE.get(intent, INTENT_GUIDANCE["FAQ"])]
    if guidance := booking_guidance(state):
        parts.append(guidance)
    if context := state.get("context"):
        parts.append(f"Relevant business information:\n{context}")

    response = llm.invoke([SystemMessage(content="\n\n".join(parts)), *recent_messages(state["messages"])])
    reply = spoken(response.text)
    logger.debug("Reply: %s", reply)
    return _with_pruning(state, AIMessage(content=reply))
