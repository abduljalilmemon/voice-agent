from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes.intent import classify_intent
from app.agent.nodes.responder import generate_response

def build_graph():
    graph = StateGraph(AgentState)

    # add nodes
    graph.add_node("intent_classifier", classify_intent)
    graph.add_node("response_generator", generate_response)

    # define flow
    graph.set_entry_point("intent_classifier")
    graph.add_edge("intent_classifier", "response_generator")
    graph.add_edge("response_generator", END)

    return graph.compile()

# single shared instance
agent = build_graph()