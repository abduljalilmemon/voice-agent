from app.agent.state import AgentState
from app.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

llm = get_llm()

INTENT_PROMPT = """You are an intent classifier for a business voice assistant.
The business ONLY handles appointments, services, pricing, hours, and location.

Classify the user message into exactly one of these intents:

- GREETING       -> hello, hi, greetings, introducing themselves, small talk
- FAQ            -> questions DIRECTLY about the business (hours, pricing, services, location)
- BOOK           -> wants to book or schedule an appointment
- COMPLAINT      -> expressing frustration or a problem with the business
- ESCALATE       -> explicitly asks for a human agent
- CONVERSATION   -> asking about something said earlier in this conversation
- OUT_OF_SCOPE   -> ANYTHING not related to this business (general knowledge, weather, math, geography etc.)

Examples:
"Hi there"                          -> GREETING
"My name is Abdul"                  -> GREETING
"What are your hours?"              -> FAQ
"Do you accept Jubilee Insurance?"  -> FAQ
"I want to book Monday"             -> BOOK
"This service was terrible"         -> COMPLAINT
"Let me talk to someone"            -> ESCALATE
"What did I just say?"              -> CONVERSATION
"What is the capital of France?"    -> OUT_OF_SCOPE
"How do I cook pasta?"              -> OUT_OF_SCOPE

Reply with ONLY the intent word. Nothing else. No explanation."""

def classify_intent(state: AgentState) -> AgentState:
    last_message = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content=INTENT_PROMPT),
        HumanMessage(content=last_message)
    ])

    intent = response.content.strip().upper()

    # safety fallback
    valid = {"FAQ", "BOOK", "COMPLAINT", "ESCALATE", "OUT_OF_SCOPE", "GREETING", "CONVERSATION"}
    if intent not in valid:
        intent = "FAQ"

    print(f"  [IntentClassifier] -> {intent}")
    return {"intent": intent}