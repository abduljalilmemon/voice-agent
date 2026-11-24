import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.llm import get_llm

logger = logging.getLogger(__name__)

# classification should be deterministic
llm = get_llm(temperature=0)

INTENTS = ["GREETING", "FAQ", "BOOK", "COLLECT", "COMPLAINT", "ESCALATE", "CONVERSATION", "OUT_OF_SCOPE"]

# how many previous messages the classifier sees (2 exchanges is enough to read short answers)
HISTORY_WINDOW = 4

INTENT_PROMPT = """You classify the LATEST user message for a service business's voice assistant
(appointments, services, pricing, hours, location).

You are given the recent conversation for context. Classify only the latest user message,
using the conversation to interpret short answers.

GREETING     = hello, introductions, thanks, small talk, closing the chat ("no, that's all")
FAQ          = a question about the business: services, prices, hours, location, insurance, staff
BOOK         = starts a new booking request
COLLECT      = answers the agent's booking question: name, phone, email, date, time, service, yes/no confirmation
COMPLAINT    = frustration or a problem with the business
ESCALATE     = asks for a human
CONVERSATION = asks about something said earlier in this chat
OUT_OF_SCOPE = anything unrelated to the business

A short answer is COLLECT only if the agent just asked for booking details.
A business question asked in the middle of a booking is still FAQ.
Reply with ONLY the intent word."""

# (previous agent message or None, user message, intent)
FEW_SHOT_EXAMPLES = [
    (None, "hi there", "GREETING"),
    (None, "what are your hours?", "FAQ"),
    (None, "i want to book an appointment", "BOOK"),
    ("Could I get your full name?", "Abdul Jalil", "COLLECT"),
    ("What date and time works for you?", "tomorrow at 3pm", "COLLECT"),
    ("Could I get your phone number?", "03477733113", "COLLECT"),
    ("Shall I go ahead and book that for you?", "yes", "COLLECT"),
    ("Is there anything else I can help you with?", "no that's it, thanks", "GREETING"),
    ("Could I get your phone number?", "how much is a cleaning?", "FAQ"),
    ("Could I get your full name?", "what's the capital of france?", "OUT_OF_SCOPE"),
    (None, "this is frustrating, nobody called me back", "COMPLAINT"),
    (None, "let me talk to a human", "ESCALATE"),
    ("Nice to meet you, Ali! How can I help?", "what is my name?", "CONVERSATION"),
]

def _speaker(message) -> str:
    return "User" if isinstance(message, HumanMessage) else "Agent"

def _format_input(history: list[str], user_message: str) -> str:
    conversation = "\n".join(history) if history else "(start of conversation)"
    return f"Recent conversation:\n{conversation}\n\nLatest user message: {user_message}"

def _few_shot_messages() -> list:
    messages = []
    for agent_line, user_line, label in FEW_SHOT_EXAMPLES:
        history = [f"Agent: {agent_line}"] if agent_line else []
        messages.append(HumanMessage(content=_format_input(history, user_line)))
        messages.append(AIMessage(content=label))
    return messages

FEW_SHOT = _few_shot_messages()

def _parse_intent(raw: str) -> str | None:
    # tolerate "Intent: COLLECT", "collect." etc. -- take the first valid label mentioned
    for token in re.findall(r"[A-Z_]+", raw.upper()):
        if token in INTENTS:
            return token
    return None

def classify_intent(state: AgentState) -> AgentState:
    *previous, latest = state["messages"]
    history = [f"{_speaker(m)}: {m.content}" for m in previous[-HISTORY_WINDOW:]]

    response = llm.invoke([
        SystemMessage(content=INTENT_PROMPT),
        *FEW_SHOT,
        HumanMessage(content=_format_input(history, latest.content))
    ])

    # safety fallback
    intent = _parse_intent(response.text) or "FAQ"

    logger.info("Intent: %s", intent)
    return {"intent": intent}
