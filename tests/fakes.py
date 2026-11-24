from langchain_core.messages import AIMessage


class FakeLLM:
    """Returns canned responses in order and records every prompt it was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("FakeLLM was called more times than the test expected")
        return AIMessage(content=self.responses.pop(0))
