import json
import logging
import sqlite3
from datetime import datetime, time

from app.booking_rules import SLOT
from app.config import storage_path

logger = logging.getLogger(__name__)

DB_PATH = storage_path("bookings.db")

def init_bookings_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            business    TEXT,
            name        TEXT,
            phone       TEXT,
            email       TEXT,
            date        TEXT,
            time        TEXT,
            service     TEXT,
            raw_data    TEXT,
            status      TEXT DEFAULT 'confirmed',
            created_at  TEXT
        )
    """)
    conn.commit()
    conn.close()

def _overlaps(start: str, other: str) -> bool:
    a, b = time.fromisoformat(start), time.fromisoformat(other)
    gap = abs((a.hour * 60 + a.minute) - (b.hour * 60 + b.minute))
    return gap < SLOT.total_seconds() / 60

def _slot_taken(conn: sqlite3.Connection, date_iso: str, time_iso: str) -> bool:
    rows = conn.execute("SELECT time FROM bookings WHERE date = ? AND status = 'confirmed'", (date_iso,)).fetchall()
    taken = []
    for (other,) in rows:
        try:
            taken.append(_overlaps(time_iso, other))
        except (TypeError, ValueError):  # bookings made before dates were validated
            continue
    return any(taken)

def is_slot_taken(date_iso: str, time_iso: str) -> bool:
    init_bookings_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        return _slot_taken(conn, date_iso, time_iso)
    finally:
        conn.close()

def save_booking(session_id: str, business_name: str, booking_data: dict) -> int | None:
    """Save the booking, or return None if its slot was taken in the meantime.

    The check and the insert run in one write-locked transaction, so two callers confirming the
    same slot at the same moment can't both get it.
    """
    init_bookings_db()
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    conn.execute("BEGIN IMMEDIATE")
    if _slot_taken(conn, booking_data["date"], booking_data["time"]):
        conn.execute("ROLLBACK")
        conn.close()
        logger.info("Slot %s %s was taken before saving", booking_data["date"], booking_data["time"])
        return None
    cursor = conn.execute(
        """INSERT INTO bookings
           (session_id, business, name, phone, email, date, time, service, raw_data, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            business_name,
            booking_data.get("name"),
            booking_data.get("phone"),
            booking_data.get("email"),
            booking_data.get("date"),
            booking_data.get("time"),
            booking_data.get("service"),
            json.dumps(booking_data),
            datetime.now().isoformat()
        )
    )
    booking_id = cursor.lastrowid
    conn.execute("COMMIT")
    conn.close()
    logger.info("Saved booking #%s", booking_id)
    return booking_id

def get_all_bookings() -> list[dict]:
    """All bookings, newest first, as dicts keyed by column name."""
    init_bookings_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM bookings ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]