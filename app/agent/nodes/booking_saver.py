from app.agent.state import AgentState
from app.config import BIZ_CONFIG, OWNER_EMAIL
from app.integrations.booking_store import save_booking
from app.integrations.notifications import send_booking_email


def save_completed_booking(state: AgentState) -> AgentState:
    """Persist a completed booking exactly once per conversation and notify the owner."""
    if state.get("booking_id"):
        return {}

    booking_data = state.get("booking_data") or {}
    if not booking_data.get("complete"):
        return {}

    booking_id = save_booking(
        state["session_id"],
        BIZ_CONFIG["business_name"],
        booking_data
    )
    if booking_id is None:  # someone else took this slot a moment ago
        rest = {k: v for k, v in booking_data.items() if k not in ("date", "time")}
        return {"booking_data": {**rest, "complete": False},
                "booking_issue": "Sorry, someone just booked that time."}
    send_booking_email(
        booking_data,
        BIZ_CONFIG["business_name"],
        OWNER_EMAIL
    )
    return {"booking_id": booking_id, "new_booking_id": booking_id}
