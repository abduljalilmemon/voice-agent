"""Intent classifier eval -- calls the real LLM configured in .env.

Usage (from repo root):  python evals/intent_eval.py [runs_per_case]

Results with groq / openai/gpt-oss-20b, 3 runs per case:
  last-message-only classifier : 44/54 = 81%
  context-aware classifier     : 54/54 = 100%
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes.intent import classify_intent

A, U = AIMessage, HumanMessage
CASES = [
 ([], "hi there", {"GREETING"}),
 ([], "what are your hours?", {"FAQ"}),
 ([], "I'd like to book a cleaning", {"BOOK"}),
 ([A("Could I get your full name?")], "Sara Khan", {"COLLECT"}),
 ([A("What date and time works for you?")], "friday morning", {"COLLECT"}),
 ([A("Could I get your phone number?")], "0300 1234567", {"COLLECT"}),
 ([A("Just to confirm: cleaning on Friday at 10am. Shall I book it?")], "yes please", {"COLLECT"}),
 ([A("What date works for you?")], "the 14th", {"COLLECT"}),
 ([A("Is there anything else I can help you with?")], "no", {"GREETING", "CONVERSATION"}),
 ([A("We're open 9am to 6pm, Monday to Saturday.")], "great, thanks", {"GREETING"}),
 ([A("Could I get your phone number?")], "how much does a root canal cost?", {"FAQ"}),
 ([A("What date and time works for you?")], "actually do you take Jubilee insurance?", {"FAQ"}),
 ([A("Could I get your full name?")], "what's the capital of france", {"OUT_OF_SCOPE"}),
 ([], "write me a poem about the ocean", {"OUT_OF_SCOPE"}),
 ([], "I've been waiting 40 minutes, this is ridiculous", {"COMPLAINT"}),
 ([], "can I speak to a real person", {"ESCALATE"}),
 ([U("my name is Ali"), A("Nice to meet you, Ali! How can I help?")], "what's my name?", {"CONVERSATION"}),
 ([], "do you guys do teeth whitening", {"FAQ"}),
]
runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ok = total = 0
for hist, msg, exp in CASES:
    got = []
    for _ in range(runs):
        got.append(classify_intent({"messages": hist + [U(msg)]})["intent"])
        time.sleep(0.3)
    hits = sum(g in exp for g in got)
    ok += hits
    total += runs
    mark = "PASS" if hits == runs else "FAIL"
    print(f"{mark} {msg!r:45} expected={'/'.join(sorted(exp)):22} got={got}")
print(f"\naccuracy: {ok}/{total} = {ok/total:.0%}")
