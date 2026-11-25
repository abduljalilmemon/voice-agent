"""Conversation eval -- scripted multi-turn chats through the full agent with the real LLM + knowledge base.

Each turn has a programmatic check on the reply. Bookings are stubbed (no DB writes, no email).

Usage (from repo root):  python evals/conversation_eval.py [runs_per_conversation]
On Groq's free tier (8K tokens/minute) set EVAL_TURN_DELAY=25 to stay under the rate limit.
"""
import os
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groq import RateLimitError  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

import app.agent.nodes.booking_saver as saver_node  # noqa: E402
from app.agent.graph import build_graph  # noqa: E402

saver_node.save_booking = lambda session_id, business, data: 42
saver_node.send_booking_email = lambda *args: True

# a reply that tells the caller the booking is done
CLAIMS_BOOKED = re.compile(r"\b(confirmed|all set|(is|are|'re) (booked|scheduled)|have booked|'ve booked|booking is set)\b", re.I)
# replies are spoken aloud -- markdown gets read out as noise
MARKDOWN = re.compile(r"\*\*|^\s*[-*•] |^\s*\d+\. |^#", re.M)


def has(*words):
    return lambda reply, result: all(w.lower() in reply.lower() for w in words)

def lacks(*words):
    return lambda reply, result: not any(w.lower() in reply.lower() for w in words)

def not_booked_yet(reply, result):
    """Before the booking is saved, the reply must not claim it is booked."""
    claims = CLAIMS_BOOKED.search(reply.replace("’", "'"))  # models often use curly apostrophes
    return result.get("new_booking_id") is None and not claims

def asks_for_phone(reply, result):
    return not_booked_yet(reply, result) and ("phone" in reply.lower() or "number" in reply.lower())

def saved_and_confirmed(reply, result):
    """The booking was really saved this turn and the reply repeats the details back."""
    return result.get("new_booking_id") is not None and "friday" in reply.lower()

def both(*checks):
    return lambda reply, result: all(c(reply, result) for c in checks)

ANY = lambda reply, result: True  # noqa: E731
TURN_DELAY = float(os.getenv("EVAL_TURN_DELAY", "0.5"))  # seconds between turns

CONVERSATIONS = {
    "remembers the caller's name": [
        ("hi, my name is Zara", ANY),
        ("what's my name?", has("zara")),
    ],
    "knows which business it is": [
        ("hello, who am I speaking to?", has("city dental")),
    ],
    "booking: never claims booked early, confirms when saved": [
        ("I'd like to book a cleaning", not_booked_yet),
        ("Sara Khan", not_booked_yet),
        ("this Friday at 10am", asks_for_phone),
        ("0300 1234567", saved_and_confirmed),
    ],
    "escalation is honest and gives a way to reach a human": [
        ("can I talk to a real person please?", both(lacks("please hold", "connect you"), has("1234567"))),
    ],
    "answers from the knowledge base": [
        ("how much is teeth whitening?", lambda r, _: "8,000" in r or "8000" in r),
        ("is there parking?", has("free")),
    ],
    "answers a price question mid-booking": [
        ("I want to book an appointment", ANY),
        ("how much is a root canal?", lambda r, _: "12,000" in r or "12000" in r),
    ],
    "refuses off-topic questions": [
        ("what's the capital of France?", lacks("paris")),
    ],
}


def run_turn(agent, session, text, attempts=4):
    """One turn. Free-tier rate limits (429) are quota, not agent quality: wait and resume.

    Resuming with invoke(None) continues the failed run from its checkpoint, so the user's
    message isn't added to the conversation twice.
    """
    config = {"configurable": {"thread_id": session}}
    payload = {"messages": [HumanMessage(content=text)], "intent": "", "context": "",
               "session_id": session, "new_booking_id": None}
    for attempt in range(attempts):
        try:
            return agent.invoke(payload, config=config)
        except RateLimitError:
            if attempt == attempts - 1:
                raise
            time.sleep(65)  # free-tier limits are per minute
            payload = None


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    agent = build_graph(checkpointer=MemorySaver())
    passed = total = 0
    spoken_ok = spoken_total = 0
    for name, turns in CONVERSATIONS.items():
        for run in range(runs):
            session = str(uuid.uuid4())
            failures = []
            for text, check in turns:
                try:
                    result = run_turn(agent, session, text)
                except Exception as e:  # the user would see an error -- count it as a failed check
                    total += 1
                    failures.append(f"    user: {text}\n    CRASH: {type(e).__name__}: {str(e)[:120]}")
                    break
                reply = result["messages"][-1].content
                spoken_total += 1
                if MARKDOWN.search(reply):
                    failures.append(f"    user: {text}\n    MARKDOWN in a spoken reply: {reply[:120]!r}")
                else:
                    spoken_ok += 1
                if check is not ANY:
                    total += 1
                    if check(reply, result):
                        passed += 1
                    else:
                        failures.append(f"    user: {text}\n    agent [{result['intent']}]: {reply}")
                time.sleep(TURN_DELAY)
            print(f"{'PASS' if not failures else 'FAIL'} {name} (run {run + 1})")
            for f in failures:
                print(f)
    print(f"\nchecks passed: {passed}/{total} = {passed / total:.0%}")
    print(f"voice-friendly replies (no markdown): {spoken_ok}/{spoken_total} = {spoken_ok / spoken_total:.0%}")


if __name__ == "__main__":
    main()
