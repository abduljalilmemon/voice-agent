import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

def send_booking_email(booking_data: dict, business_name: str, owner_email: str) -> bool:
    """Send booking confirmation email to business owner."""

    gmail_user = os.getenv("GMAIL_USER")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")

    if not gmail_user or not gmail_password:
        print("  [Email] -> skipped, no Gmail credentials in .env")
        return False

    # build table rows from booking data
    fields_html = ""
    field_labels = {
        "name": "Customer Name",
        "phone": "Phone Number",
        "email": "Email Address",
        "date": "Appointment Date",
        "time": "Appointment Time",
        "service": "Service Requested"
    }

    for key, label in field_labels.items():
        value = booking_data.get(key)
        if value and value not in (None, "null", ""):
            fields_html += f"""
            <tr>
                <td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;
                           font-weight:bold;color:#374151;width:40%;'>{label}</td>
                <td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;
                           color:#1a1a1a;'>{value}</td>
            </tr>"""

    html = f"""
    <html>
    <body style='font-family:Arial,sans-serif;max-width:600px;margin:0 auto;
                 background:#f9fafb;padding:20px;'>

        <div style='background:#1F4E79;padding:24px;border-radius:8px 8px 0 0;'>
            <h2 style='color:white;margin:0;font-size:20px;'>
                New Booking — {business_name}
            </h2>
            <p style='color:#BDD7EE;margin:8px 0 0 0;font-size:14px;'>
                A new appointment was booked via your AI Voice Agent
            </p>
        </div>

        <div style='background:white;padding:24px;border:1px solid #e5e7eb;
                    border-top:none;border-radius:0 0 8px 8px;'>

            <table style='width:100%;border-collapse:collapse;'>
                {fields_html}
            </table>

            <div style='margin-top:24px;padding:16px;background:#f0f4f8;
                        border-radius:8px;'>
                <p style='margin:0;font-size:13px;color:#6b7280;'>
                    This booking was made through your AI Voice Agent.
                    Please follow up with the customer to confirm the appointment.
                </p>
            </div>
        </div>

        <p style='text-align:center;font-size:12px;color:#9ca3af;margin-top:16px;'>
            Powered by AI Voice Agent
        </p>

    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"New Booking: {booking_data.get('name', 'Unknown')} — {booking_data.get('date', '')} {booking_data.get('time', '')}"
    msg["From"] = gmail_user
    msg["To"] = owner_email
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_password)
            server.sendmail(gmail_user, owner_email, msg.as_string())
        print(f"  [Email] -> sent to {owner_email}")
        return True
    except Exception as e:
        print(f"  [Email] -> failed: {e}")
        return False