"""Manual check: sends a REAL booking email to OWNER_EMAIL using the Gmail credentials in .env.

Run from the repo root:  python scripts/send_test_email.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

from app.integrations.notifications import send_booking_email

test_booking = {
    "name": "Admin Test",
    "phone": "03333333333",
    "email": "admin@gmail.com",
    "date": "27 Jan 2026",
    "time": "6 PM",
    "service": "Dental Cleaning"
}

if __name__ == "__main__":
    result = send_booking_email(
        test_booking,
        "City Dental Clinic",
        os.getenv("OWNER_EMAIL")
    )

    print(f"Email sent: {result}")
