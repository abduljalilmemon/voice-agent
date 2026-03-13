from app.llm import get_llm
from app.agent.graph import agent
from langchain_core.messages import HumanMessage
import uuid


def test_llm():
    print("Testing LLM connection...")
    llm = get_llm()
    response = llm.invoke("You are a helpful assistant. Say hello in one sentence.")
    print(f"LLM Response: {response.content}")
    print("LLM connection successful!")


def chat(message: str, session_id: str):
    print(f"\nUser: {message}")
    result = agent.invoke({
        "messages": [HumanMessage(content=message)],
        "intent": "",
        "context": "",
        "session_id": session_id
    })
    reply = result["messages"][-1].content
    print(f"Agent: {reply}")
    return reply

if __name__ == "__main__":
    session = str(uuid.uuid4())
    print("Voice Agent Test -- type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        chat(user_input, session)