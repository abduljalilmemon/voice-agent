from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from app.agent.state import AgentState
from app.agent.nodes.intent import classify_intent
from app.agent.nodes.retriever import knowledge_retriever
from app.agent.nodes.responder import generate_response

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent_classifier", classify_intent)
    graph.add_node("knowledge_retriever", knowledge_retriever)
    graph.add_node("response_generator", generate_response)

    graph.set_entry_point("intent_classifier")
    
    graph.add_edge("intent_classifier", "knowledge_retriever")
    graph.add_edge("knowledge_retriever", "response_generator")
    graph.add_edge("response_generator", END)

    conn = sqlite3.connect("memory.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    return graph.compile(checkpointer=memory)

agent = build_graph()