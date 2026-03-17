from app.agent.state import AgentState
from app.llm import get_llm
from langchain_core.messages import SystemMessage, AIMessage

llm = get_llm()

SYSTEM_PROMPT = """You are a friendly, professional voice assistant for a business.
Keep responses concise and natural -- this will be spoken aloud.
Maximum 2-3 sentences per response.
Never use bullet points or markdown.
Never answer questions unrelated to the business under any circumstances."""

STATIC_RESPONSES = {
    "OUT_OF_SCOPE": "I can only help with questions about our business, such as services, hours, pricing, or bookings. Is there anything like that I can help you with?",
    "ESCALATE": "Of course, let me connect you with a team member right away. Please hold for a moment."
}

def generate_response(state: AgentState) -> AgentState:
    intent = state.get("intent", "FAQ")
    context = state.get("context", "")

    # return static response immediately -- no LLM call
    if intent in STATIC_RESPONSES:
        reply = STATIC_RESPONSES[intent]
        print(f"  [ResponseGenerator] -> (static) {reply}...")
        return {"messages": [AIMessage(content=reply)]}
    
    if intent in {"CONVERSATION", "GREETING", "COLLECT"}:
        system_content = """You are a friendly voice assistant for a service business.
    If the user is providing booking information, acknowledge it warmly and ask 
    for the next missing piece of information in this order:
    1. Full name (if missing)
    2. Preferred date and time (if missing)
    3. Phone number (if missing)
    Only ask for ONE thing at a time. Check conversation history before asking."""
        response = llm.invoke([
            SystemMessage(content=system_content),
            *state["messages"]
        ])
        print(f"  [ResponseGenerator] -> {response.content[:60]}...")
        return {"messages": [AIMessage(content=response.content)]}


    system_content = SYSTEM_PROMPT
    if context:
        system_content += f"\n\nRelevant business information:\n{context}"
    if intent == "COMPLAINT":
        system_content += "\n\nThe user is frustrated. Be extra empathetic and apologetic."
    elif intent == "BOOK":
        system_content += """
    The user wants to book an appointment. Collect information in this order:
    1. First ask for their full name (if not already provided)
    2. Then ask for preferred date and time (if not already provided)
    3. Then ask for phone number (if not already provided)
    Only ask for ONE piece of missing information per response.
    Never ask for everything at once.
    Check the conversation history before asking -- don't ask for info already given."""
    response = llm.invoke([
        SystemMessage(content=system_content),
        *state["messages"]
    ])

    print(f"  [ResponseGenerator] -> {response.content}...")
    return {"messages": [AIMessage(content=response.content)]}