import functools
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from app.agent.nodes.booking_saver import save_completed_booking
from app.agent.nodes.responder import generate_response
from app.agent.nodes.retriever import knowledge_retriever
from app.agent.nodes.understand import understand
from app.agent.state import AgentState
from app.config import storage_path
from app.metrics import metrics


def _timed(stage: str, node):
    """Record each step's duration as agent.<stage> in the latency metrics."""
    @functools.wraps(node)
    def timed_node(state: AgentState) -> AgentState:
        with metrics.timer(f"agent.{stage}"):
            return node(state)
    return timed_node

def route_after_retrieval(state: AgentState) -> str:
    # save once: never again after a booking_id exists
    if (state.get("booking_data") or {}).get("complete") and not state.get("booking_id"):
        return "booking_saver"
    return "response_generator"

def build_graph(checkpointer=None):
    """understand (intent + booking details, one LLM call) -> knowledge -> [save] -> reply.

    Booking work happens before the reply, so the reply knows exactly what has been collected and
    whether the booking was saved -- it never has to guess whether to say "you're booked".
    """
    graph = StateGraph(AgentState)

    graph.add_node("understand", _timed("understand", understand))
    graph.add_node("knowledge_retriever", _timed("knowledge", knowledge_retriever))
    graph.add_node("booking_saver", _timed("save_booking", save_completed_booking))
    graph.add_node("response_generator", _timed("reply", generate_response))

    graph.set_entry_point("understand")
    graph.add_edge("understand", "knowledge_retriever")
    graph.add_conditional_edges("knowledge_retriever", route_after_retrieval,
                                ["booking_saver", "response_generator"])
    graph.add_edge("booking_saver", "response_generator")
    graph.add_edge("response_generator", END)

    if checkpointer is None:
        conn = sqlite3.connect(storage_path("memory.db"), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)

agent = build_graph()
