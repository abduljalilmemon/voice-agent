import json
import logging
import sqlite3
from datetime import datetime

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

def save_booking(session_id: str, business_name: str, booking_data: dict) -> int:
    init_bookings_db()
    conn = sqlite3.connect(DB_PATH)
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
    conn.commit()
    conn.close()
    logger.info("Saved booking #%s", booking_id)
    return booking_id

def get_all_bookings() -> list:
    init_bookings_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT * FROM bookings ORDER BY created_at DESC").fetchall()
    conn.close()
    return rows