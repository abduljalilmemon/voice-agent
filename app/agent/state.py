from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages] # full conversation history
    intent: str # classified intent
    context: str # retrieved knowledge base info
    session_id: str # unique per conversation
    booking_data: dict # store booking info like name, date, time etc.