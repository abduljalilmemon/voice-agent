from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from app.agent.state import AgentState
from app.agent.nodes.intent import classify_intent
from app.agent.nodes.retriever import knowledge_retriever
from app.agent.nodes.responder import generate_response
from app.agent.nodes.booking_extractor import extract_booking

def route_after_response(state: AgentState) -> str:
    if state.get("intent") in {"BOOK", "COLLECT"}:
        return "booking_extractor"
    return END

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent_classifier", classify_intent)
    graph.add_node("knowledge_retriever", knowledge_retriever)
    graph.add_node("response_generator", generate_response)
    graph.add_node("booking_extractor", extract_booking)

    graph.set_entry_point("intent_classifier")
    graph.add_edge("intent_classifier", "knowledge_retriever")
    graph.add_edge("knowledge_retriever", "response_generator")

    graph.add_conditional_edges(
        "response_generator",
        route_after_response,
        {
            "booking_extractor": "booking_extractor",
            END: END
        }
    )
    graph.add_edge("booking_extractor", END)

    conn = sqlite3.connect("memory.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    return graph.compile(checkpointer=memory)

agent = build_graph()