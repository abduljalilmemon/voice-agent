import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from app.agent.nodes.booking_extractor import extract_booking
from app.agent.nodes.booking_saver import save_completed_booking
from app.agent.nodes.intent import classify_intent
from app.agent.nodes.responder import generate_response
from app.agent.nodes.retriever import knowledge_retriever
from app.agent.state import AgentState
from app.config import storage_path


def route_after_retrieval(state: AgentState) -> str:
    # once a booking is saved, stop extracting -- prevents duplicate saves
    if state.get("booking_id"):
        return "response_generator"
    if state.get("intent") in {"BOOK", "COLLECT"}:
        return "booking_extractor"
    return "response_generator"

def route_after_extraction(state: AgentState) -> str:
    if (state.get("booking_data") or {}).get("complete"):
        return "booking_saver"
    return "response_generator"

def build_graph(checkpointer=None):
    """intent -> knowledge -> [booking extraction -> save] -> reply.

    Booking work runs before the reply, so the reply knows exactly what has been collected and
    whether the booking was saved -- it never has to guess whether to say "you're booked".
    """
    graph = StateGraph(AgentState)

    graph.add_node("intent_classifier", classify_intent)
    graph.add_node("knowledge_retriever", knowledge_retriever)
    graph.add_node("booking_extractor", extract_booking)
    graph.add_node("booking_saver", save_completed_booking)
    graph.add_node("response_generator", generate_response)

    graph.set_entry_point("intent_classifier")
    graph.add_edge("intent_classifier", "knowledge_retriever")
    graph.add_conditional_edges("knowledge_retriever", route_after_retrieval,
                                ["booking_extractor", "response_generator"])
    graph.add_conditional_edges("booking_extractor", route_after_extraction,
                                ["booking_saver", "response_generator"])
    graph.add_edge("booking_saver", "response_generator")
    graph.add_edge("response_generator", END)

    if checkpointer is None:
        conn = sqlite3.connect(storage_path("memory.db"), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)

agent = build_graph()
