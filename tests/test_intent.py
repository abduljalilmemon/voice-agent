import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agent.nodes.intent as intent_node
from tests.fakes import FakeLLM


@pytest.mark.parametrize("raw, expected", [
    ("COLLECT", "COLLECT"),
    ("collect.", "COLLECT"),
    ("Intent: OUT_OF_SCOPE", "OUT_OF_SCOPE"),
    ("  faq\n", "FAQ"),
    ("I think this is a booking", None),
    ("", None),
])
def test_parse_intent(raw, expected):
    assert intent_node._parse_intent(raw) == expected


def _classify(monkeypatch, messages, llm_output="COLLECT"):
    fake = FakeLLM([llm_output])
    monkeypatch.setattr(intent_node, "llm", fake)
    return intent_node.classify_intent({"messages": messages}), fake.calls[0]


def test_classifier_sees_recent_history_and_few_shot_examples(monkeypatch):
    history = [HumanMessage(content=f"user {i}") if i % 2 == 0 else AIMessage(content=f"agent {i}")
               for i in range(6)]
    result, prompt = _classify(monkeypatch, history + [HumanMessage(content="Sara Khan")])

    assert result == {"intent": "COLLECT"}
    assert len(prompt) == 1 + 2 * len(intent_node.FEW_SHOT_EXAMPLES) + 1  # system + examples + input
    final_input = prompt[-1].content
    assert "Latest user message: Sara Khan" in final_input
    assert "Agent: agent 5" in final_input and "User: user 2" in final_input  # last 4 messages
    assert "user 0" not in final_input and "agent 1" not in final_input   # older ones dropped


def test_first_message_has_no_history(monkeypatch):
    _, prompt = _classify(monkeypatch, [HumanMessage(content="hi")], "GREETING")
    assert "(start of conversation)" in prompt[-1].content


def test_unrecognised_output_falls_back_to_faq(monkeypatch):
    result, _ = _classify(monkeypatch, [HumanMessage(content="hmm")], "no idea")
    assert result == {"intent": "FAQ"}
