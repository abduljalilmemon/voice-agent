import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.llm import get_llm

logger = logging.getLogger(__name__)

llm = get_llm(temperature=0)  # extraction should be deterministic

EXTRACT_PROMPT = """You are a data extractor. Extract booking information from the conversation.

Return ONLY a valid JSON object with these fields (use null if not mentioned):
{
    "name": "customer full name or null",
    "phone": "phone number or null",
    "email": "email address or null",
    "date": "appointment date or null",
    "time": "appointment time or null",
    "service": "service requested or null",
    "complete": true or false
}

Set "complete" to true when name, date, time, AND phone are all present.
Email is optional -- do NOT require it for complete=true.
Return ONLY the JSON. No explanation. No markdown. No backticks."""

def extract_booking(state: AgentState) -> AgentState:
    intent = state.get("intent", "")

    # only run on booking-related intents -- return {} so previously collected data is kept
    if intent not in {"BOOK", "COLLECT"}:
        return {}

    # build full conversation text
    conversation = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in state["messages"]
    ])

    response = llm.invoke([
        SystemMessage(content=EXTRACT_PROMPT),
        HumanMessage(content=f"Extract booking info from this conversation:\n\n{conversation}")
    ])

    try:
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        booking_data = json.loads(raw)

        # clean string "null" values to actual None
        booking_data = {
            k: (None if v == "null" or v == "" else v)
            for k, v in booking_data.items()
        }

        # calculate complete ourselves -- never trust the LLM for this
        required = ["name", "date", "time", "phone"]
        booking_data["complete"] = all(
            booking_data.get(field) not in (None, "null", "")
            for field in required
        )

        logger.info("Booking extraction: complete=%s", booking_data["complete"])
        logger.debug("Extracted booking data: %s", booking_data)
        return {"booking_data": booking_data}

    except Exception as e:
        logger.warning("Could not parse booking extraction: %s", e)
        logger.debug("Raw extractor output: %s", response.text)
        return {}  # keep previously collected data