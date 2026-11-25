from langchain_core.messages import AIMessage

from app.agent.nodes.understand import BookingDetails, TurnUnderstanding


class FakeLLM:
    """Returns canned responses in order and records every prompt it was sent.

    Strings come back as chat messages; anything else (e.g. a structured-output object) as is.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("FakeLLM was called more times than the test expected")
        response = self.responses.pop(0)
        return AIMessage(content=response) if isinstance(response, str) else response


def understood(intent, **booking):
    """What the understand node's LLM returns: an intent plus booking details from the latest message."""
    fields = {field: None for field in BookingDetails.model_fields}
    return TurnUnderstanding(intent=intent, booking=BookingDetails(**{**fields, **booking}))
