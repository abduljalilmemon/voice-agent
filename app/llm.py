import os

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError

load_dotenv()

# per-call limit: replies normally take under a second, and retries wrap each call
LLM_TIMEOUT_SECONDS = 20

# Default model per provider -- override any of them with LLM_MODEL in .env
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "groq": "openai/gpt-oss-20b",
}

def get_llm(temperature: float = 0.7, schema: type[BaseModel] | None = None, reasoning_effort: str | None = None):
    """Chat model for the configured provider.

    schema: invoke() returns a validated instance of this Pydantic model (native JSON-schema output).
    reasoning_effort: for Groq's gpt-oss reasoning models ("low" | "medium" | "high"); ignored elsewhere.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown LLM provider: {provider}")
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # Current Claude models reject non-default temperature, so it is not passed.
        # Low effort keeps thinking short -- replies are spoken, so latency matters most.
        llm = ChatAnthropic(model=model, effort=os.getenv("ANTHROPIC_EFFORT", "low"), timeout=LLM_TIMEOUT_SECONDS)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, temperature=temperature, timeout=LLM_TIMEOUT_SECONDS)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, timeout=LLM_TIMEOUT_SECONDS)
    else:  # groq
        from langchain_groq import ChatGroq
        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort and "gpt-oss" in model else {}
        llm = ChatGroq(model=model, temperature=temperature, groq_api_key=os.getenv("GROQ_API_KEY"),
                       timeout=LLM_TIMEOUT_SECONDS, **extra)

    retry_on: tuple[type[Exception], ...] = ()
    if schema is not None:
        llm = llm.with_structured_output(schema, method="json_schema")
        retry_on += (OutputParserException, ValidationError)  # a malformed object is worth one more try
    if provider == "groq":
        from groq import BadRequestError
        # gpt-oss on Groq occasionally emits a malformed tool call instead of text, which the API
        # rejects with 400 "tool_use_failed". It's sampling noise, so a retry almost always succeeds.
        retry_on += (BadRequestError,)
    if retry_on:
        llm = llm.with_retry(retry_if_exception_type=retry_on, stop_after_attempt=3, wait_exponential_jitter=False)
    if provider == "groq":
        from groq import RateLimitError
        # free-tier limits are per minute: back off briefly before giving up (~1s, then ~2-4s)
        llm = llm.with_retry(retry_if_exception_type=(RateLimitError,), stop_after_attempt=3,
                             exponential_jitter_params={"initial": 1, "max": 6})
    return llm

def is_rate_limited(error: BaseException) -> bool:
    """True for a provider's "too many requests" error (Groq, Anthropic and OpenAI all name it so)."""
    return type(error).__name__ == "RateLimitError"
