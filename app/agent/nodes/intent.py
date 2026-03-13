from app.agent.state import AgentState
from app.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

llm = get_llm()

INTENT_PROMPT = """You are an intent classifier for a business voice assistant.
The business ONLY handles appointments, services, pricing, hours, and location.

Classify the user message into exactly one of these intents:

- FAQ           → questions DIRECTLY about the business (hours, pricing, services, location)
- BOOK          → wants to book or schedule an appointment
- COMPLAINT     → expressing frustration or a problem with the business
- ESCALATE      → explicitly asks for a human agent
- OUT_OF_SCOPE  → ANYTHING not related to this business (general knowledge, weather, math, geography, etc.)

Examples:
"What are your hours?" → FAQ
"I want to book Monday" → BOOK
"This service was terrible" → COMPLAINT
"Let me talk to someone" → ESCALATE
"What is the capital of France?" → OUT_OF_SCOPE
"How do I cook pasta?" → OUT_OF_SCOPE
"What is 2+2?" → OUT_OF_SCOPE

Reply with ONLY the intent word. Nothing else. No explanation."""

def classify_intent(state: AgentState) -> AgentState:
    last_message = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content=INTENT_PROMPT),
        HumanMessage(content=last_message)
    ])

    intent = response.content.strip().upper()

    # safety fallback
    valid = {"FAQ", "BOOK", "COMPLAINT", "ESCALATE", "OUT_OF_SCOPE"}
    if intent not in valid:
        intent = "FAQ"

    print(f"  [IntentClassifier] → {intent}")
    return {"intent": intent}