import os
from dotenv import load_dotenv

load_dotenv()

# Default model per provider -- override any of them with LLM_MODEL in .env
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "groq": "openai/gpt-oss-20b",
}

def get_llm(temperature: float = 0.7):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown LLM provider: {provider}")
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # Current Claude models reject non-default temperature, so it is not passed.
        # Low effort keeps thinking short -- replies are spoken, so latency matters most.
        return ChatAnthropic(
            model=model,
            effort=os.getenv("ANTHROPIC_EFFORT", "low")
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    else:  # groq
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
