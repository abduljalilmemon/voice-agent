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

def chat(message: str, config: dict):
    print(f"\nUser: {message}")
    
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "intent": "",
            "context": "",
            "session_id": config["configurable"]["thread_id"]
        },
        config=config
    )
    print(agent.get_state(config))

    # debug — show full message history
    print(f"  [Memory] -> {len(result['messages'])} messages in history")
    for msg in result["messages"]:
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        print(f"    {role}: {msg.content[:50]}")

    reply = result["messages"][-1].content
    print(f"Agent: {reply}")
    return reply

if __name__ == "__main__":
    # config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    config = {"configurable": {"thread_id": "0a06368d-9403-4e81-9580-ed71f33f0d16"}}

    print("Voice Agent Test -- type 'quit' to exit\n")
    print(f"Session: {config['configurable']['thread_id']}\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        chat(user_input, config)