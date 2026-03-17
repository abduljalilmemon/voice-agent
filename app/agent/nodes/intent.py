from langchain.messages import AIMessage

from app.agent.state import AgentState
from app.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

llm = get_llm()

# INTENT_PROMPT = """You are an intent classifier for a business voice assistant.
# The business ONLY handles appointments, services, pricing, hours, and location.

# Classify the user message into exactly one of these intents:

# - GREETING       -> hello, hi, greetings, introducing themselves, small talk
# - FAQ            -> questions DIRECTLY about the business (hours, pricing, services, location)
# - BOOK           -> wants to book or schedule an appointment
# - COMPLAINT      -> expressing frustration or a problem with the business
# - ESCALATE       -> explicitly asks for a human agent
# - CONVERSATION   -> asking about something said earlier in this conversation
# - OUT_OF_SCOPE   -> ANYTHING not related to this business (general knowledge, weather, math, geography etc.)

# Examples:
# "Hi there"                          -> GREETING
# "My name is Abdul"                  -> GREETING
# "What are your hours?"              -> FAQ
# "Do you accept Jubilee Insurance?"  -> FAQ
# "I want to book Monday"             -> BOOK
# "This service was terrible"         -> COMPLAINT
# "Let me talk to someone"            -> ESCALATE
# "What did I just say?"              -> CONVERSATION
# "What is the capital of France?"    -> OUT_OF_SCOPE
# "How do I cook pasta?"              -> OUT_OF_SCOPE

# Reply with ONLY the intent word. Nothing else. No explanation."""


INTENT_PROMPT = """Classify the user message into one intent:
GREETING, FAQ, BOOK, COMPLAINT, ESCALATE, CONVERSATION, COLLECT, OUT_OF_SCOPE

Business context: any service business (appointments, services, pricing, hours, location)
BOOK = first expression of wanting to book
COLLECT = ANY response during an ongoing booking conversation including:
          names, phone numbers, dates, times, yes/no confirmations, short answers
CONVERSATION = asking about something said earlier in this chat
OUT_OF_SCOPE = general knowledge questions completely unrelated to this business

When in doubt between COLLECT and OUT_OF_SCOPE, choose COLLECT.
Reply with ONLY the intent word."""



FEW_SHOT = [
    HumanMessage(content="hi there"),                    AIMessage(content="GREETING"),
    HumanMessage(content="what are your hours?"),        AIMessage(content="FAQ"),
    HumanMessage(content="book an appointment"),         AIMessage(content="BOOK"),
    HumanMessage(content="i want to book"),              AIMessage(content="BOOK"),
    HumanMessage(content="this is frustrating"),         AIMessage(content="COMPLAINT"),
    HumanMessage(content="talk to a human"),             AIMessage(content="ESCALATE"),
    HumanMessage(content="what is my name?"),            AIMessage(content="CONVERSATION"),
    HumanMessage(content="capital of france?"),          AIMessage(content="OUT_OF_SCOPE"),
    HumanMessage(content="my phone is 03477733113"),     AIMessage(content="COLLECT"),
    HumanMessage(content="03477733113"),                 AIMessage(content="COLLECT"),
    HumanMessage(content="my email is a@b.com"),         AIMessage(content="COLLECT"),
    HumanMessage(content="Abdul Jalil"),                 AIMessage(content="COLLECT"),
    HumanMessage(content="my name is Abdul"),            AIMessage(content="COLLECT"),
    HumanMessage(content="tomorrow at 3pm"),             AIMessage(content="COLLECT"),
    HumanMessage(content="6pm wednesday"),               AIMessage(content="COLLECT"),
    HumanMessage(content="yes"),                         AIMessage(content="COLLECT"),
    HumanMessage(content="no"),                          AIMessage(content="COLLECT"),
    HumanMessage(content="thats it"),                    AIMessage(content="COLLECT"),
    HumanMessage(content="confirmed"),                   AIMessage(content="COLLECT"),
]

def classify_intent(state: AgentState) -> AgentState:
    last_message = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content=INTENT_PROMPT),
        HumanMessage(content=last_message)
    ])

    intent = response.content.strip().upper()

    # safety fallback
    valid = {"FAQ", "BOOK", "COMPLAINT", "ESCALATE", "OUT_OF_SCOPE", "GREETING", "CONVERSATION", "COLLECT"}
    if intent not in valid:
        intent = "FAQ"

    print(f"  [IntentClassifier] -> {intent}")
    return {"intent": intent}