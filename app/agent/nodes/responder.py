from app.agent.state import AgentState
from app.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = get_llm()

SYSTEM_PROMPT = """You are a friendly, professional voice assistant for a business.
Keep responses concise and natural -- this will be spoken aloud.
Maximum 2-3 sentences per response.
Never use bullet points or markdown.
Never answer questions unrelated to the business under any circumstances."""

# hardcoded responses -- no LLM needed
STATIC_RESPONSES = {
    "OUT_OF_SCOPE": "I can only help with questions about our business, such as services, hours, pricing, or bookings. Is there anything like that I can help you with?",
    "ESCALATE": "Of course, let me connect you with a team member right away. Please hold for a moment.",
}

def generate_response(state: AgentState) -> AgentState:
    intent = state.get("intent", "FAQ")
    context = state.get("context", "")

    # return static response immediately -- no LLM call
    if intent in STATIC_RESPONSES:
        reply = STATIC_RESPONSES[intent]
        print(f"  [ResponseGenerator] -> (static) {reply[:60]}...")
        return {"messages": [AIMessage(content=reply)]}

    system_content = SYSTEM_PROMPT
    if context:
        system_content += f"\n\nRelevant business information:\n{context}"
    if intent == "COMPLAINT":
        system_content += "\n\nThe user is frustrated. Be extra empathetic and apologetic."
    elif intent == "BOOK":
        system_content += "\n\nThe user wants to book. Ask for their name and preferred date/time."

    response = llm.invoke([
        SystemMessage(content=system_content),
        *state["messages"]
    ])

    print(f"  [ResponseGenerator] -> {response.content[:60]}...")
    return {"messages": [AIMessage(content=response.content)]}