from app.integrations.notifications import send_booking_email

test_booking = {
    "name": "Admin Test",
    "phone": "03333333333",
    "email": "admin@gmail.com",
    "date": "27 Jan 2026",
    "time": "6 PM",
    "service": "Dental Cleaning"
}

result = send_booking_email(
    test_booking,
    "City Dental Clinic",
    "jay207803@gmail.com"
)

print(f"Email sent: {result}")