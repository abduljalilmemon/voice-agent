"""Booking rules, enforced in code: real dates, opening hours, slot length, phone and email format.

The LLM reports what the caller said ("this Friday", "10:00"); this module resolves the date and
decides whether the slot is bookable. Functions take `now` so tests don't depend on the clock.
"""
import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import dateparser

from app.config import BIZ_CONFIG

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
TIMEZONE = ZoneInfo(BIZ_CONFIG["timezone"])
SLOT = timedelta(minutes=BIZ_CONFIG["appointment_minutes"])

def now_local() -> datetime:
    return datetime.now(TIMEZONE).replace(tzinfo=None)

# ── parsing (the LLM is asked for ISO formats; anything else is rejected) ──

def parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value.strip()) if value else None
    except ValueError:
        return None

def resolve_date(phrase: str | None, now: datetime) -> date | None:
    """Turn what the caller said ("this Friday", "tomorrow", "the 2nd of December") into a date.

    Done in code, not by the LLM: the model got weekday arithmetic wrong ("next Monday" -> a Friday).
    """
    if not phrase:
        return None
    phrase = phrase.strip()
    if iso := parse_date(phrase):
        return iso
    if match := re.fullmatch(r"(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?", phrase, re.IGNORECASE):
        # "the 14th": the next time that day of the month comes round
        for months_ahead in range(3):
            year, month = divmod(now.month - 1 + months_ahead, 12)
            try:
                candidate = date(now.year + year, month + 1, int(match.group(1)))
            except ValueError:  # e.g. the 31st in a 30-day month
                continue
            if candidate >= now.date():
                return candidate
        return None
    parsed = dateparser.parse(phrase, languages=["en"], settings={
        "PREFER_DATES_FROM": "future", "RELATIVE_BASE": now, "DATE_ORDER": "DMY"})
    return parsed.date() if parsed else None

def parse_time(value: str | None) -> time | None:
    try:
        return time.fromisoformat(value.strip()) if value else None
    except ValueError:
        return None

def normalize_phone(value: str | None) -> str | None:
    """Keep digits (and a leading +); a real number has 7-15 digits (the E.164 maximum)."""
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if not 7 <= len(digits) <= 15:
        return None
    return ("+" if value.strip().startswith("+") else "") + digits

def valid_email(value: str | None) -> bool:
    return bool(value) and re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()) is not None

# ── human-friendly text ────────────────────────────────────────

def format_time(t: time) -> str:
    return t.strftime("%I:%M %p").lstrip("0").replace(":00 ", " ")

def format_date(d: date) -> str:
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B')}"

def opening_hours_text() -> str:
    """e.g. "Monday to Saturday, 9 AM to 7 PM. Closed on Sunday." -- derived, so it can't disagree."""
    hours = BIZ_CONFIG["opening_hours"]
    groups: list[list] = []  # [first_day, last_day, hours]
    for day in DAYS:
        if groups and groups[-1][2] == hours[day]:
            groups[-1][1] = day
        else:
            groups.append([day, day, hours[day]])
    parts, closed = [], []
    for first, last, span in groups:
        days = first.title() if first == last else f"{first.title()} to {last.title()}"
        if span is None:
            closed.append(days)
        else:
            parts.append(f"{days}, {format_time(time.fromisoformat(span[0]))} to {format_time(time.fromisoformat(span[1]))}")
    text = "; ".join(parts) + "."
    return text + (f" Closed on {' and '.join(closed)}." if closed else "")

# ── the rules ──────────────────────────────────────────────────

def day_problem(d: date, now: datetime) -> str | None:
    """Can anything be booked on this day at all?"""
    if d < now.date():
        return f"{format_date(d)} has already passed."
    if BIZ_CONFIG["opening_hours"][DAYS[d.weekday()]] is None:
        return f"We're closed on {DAYS[d.weekday()].title()}s."
    return None

def slot_problem(d: date, t: time, now: datetime) -> str | None:
    """Is this exact appointment slot bookable (ignoring other bookings)?"""
    if problem := day_problem(d, now):
        return problem
    start = datetime.combine(d, t)
    if start <= now:
        return "That time has already passed."
    opens, closes = (time.fromisoformat(x) for x in BIZ_CONFIG["opening_hours"][DAYS[d.weekday()]])
    if t < opens or (start + SLOT).time() > closes or (start + SLOT).date() != d:
        return f"On {DAYS[d.weekday()].title()}s we take appointments from {format_time(opens)} until {format_time(closes)}."
    return None
