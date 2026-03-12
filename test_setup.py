from app.llm import get_llm

def test_llm():
    print("Testing LLM connection...")
    llm = get_llm()
    response = llm.invoke("You are a helpful assistant. Say hello in one sentence.")
    print(f"LLM Response: {response.content}")
    print("LLM connection successful!")

if __name__ == "__main__":
    test_llm()