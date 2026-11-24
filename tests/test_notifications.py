import email
from email.header import decode_header, make_header

import pytest

from app.integrations import notifications


@pytest.fixture
def smtp(monkeypatch):
    """Fake Gmail SMTP: captures the sent message instead of sending it."""
    sent = {}

    class FakeSMTP:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def login(self, *args):
            pass

        def sendmail(self, sender, recipient, raw):
            sent.update(to=recipient, message=email.message_from_string(raw))

    monkeypatch.setattr(notifications.smtplib, "SMTP_SSL", FakeSMTP)
    monkeypatch.setenv("GMAIL_USER", "agent@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-password")
    return sent


def _subject(message):
    return str(make_header(decode_header(message["Subject"])))  # subjects are RFC 2047 encoded


def _html(message):
    return message.get_payload()[0].get_payload(decode=True).decode()


def test_sends_booking_details_to_owner(smtp):
    booking = {"name": "Sara Khan", "phone": "03001234567", "date": "Friday", "time": "10am"}
    assert notifications.send_booking_email(booking, "City Dental", "owner@example.com")
    assert smtp["to"] == "owner@example.com"
    assert _subject(smtp["message"]) == "New Booking: Sara Khan — Friday 10am"
    assert "03001234567" in _html(smtp["message"])


def test_booking_data_is_html_escaped(smtp):
    booking = {"name": "Eve", "phone": "<script>alert(1)</script>", "date": "Mon", "time": "6pm"}
    notifications.send_booking_email(booking, "<b>Clinic</b>", "owner@example.com")
    body = _html(smtp["message"])
    assert "<script>" not in body and "&lt;script&gt;" in body
    assert "<b>Clinic</b>" not in body


def test_newline_in_name_cannot_inject_headers(smtp):
    booking = {"name": "Eve\r\nBcc: victim@example.com", "date": "Mon", "time": "6pm"}
    notifications.send_booking_email(booking, "City Dental", "owner@example.com")
    assert smtp["message"]["Bcc"] is None
    assert "\n" not in _subject(smtp["message"])


def test_skipped_without_owner_email(smtp):
    assert notifications.send_booking_email({"name": "Sara"}, "City Dental", None) is False
    assert smtp == {}


def test_skipped_without_gmail_credentials(smtp, monkeypatch):
    monkeypatch.setenv("GMAIL_USER", "")
    assert notifications.send_booking_email({"name": "Sara"}, "City Dental", "owner@example.com") is False
    assert smtp == {}
