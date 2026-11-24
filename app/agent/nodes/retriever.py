import logging

from app.agent.state import AgentState
from app.knowledge.retriever import retrieve

logger = logging.getLogger(__name__)

# only retrieve for these intents
RETRIEVAL_INTENTS = {"FAQ", "BOOK", "COMPLAINT"}

def knowledge_retriever(state: AgentState) -> AgentState:
    intent = state.get("intent", "")

    if intent not in RETRIEVAL_INTENTS:
        return {"context": ""}

    last_message = state["messages"][-1].content
    context = retrieve(last_message)

    if context:
        logger.debug("Knowledge base: %d chars of context", len(context))
    else:
        logger.info("Knowledge base: no relevant context found")

    return {"context": context}