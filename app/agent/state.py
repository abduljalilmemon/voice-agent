from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages] # full conversation history
    intent: str # classified intent (per turn)
    context: str # retrieved knowledge base info (per turn)
    session_id: str # unique per conversation
    booking_data: dict # booking info like name, date, time etc. (persists across turns)
    booking_id: int | None # set once the booking is saved -- guards against duplicate saves
    new_booking_id: int | None # set only on the turn the booking was saved (reset every turn)
