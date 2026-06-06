import os
from datetime import datetime, timedelta, timezone
import httpx

CAL_API_KEY = os.getenv("CAL_API_KEY", "")
CAL_EVENT_TYPE_ID = os.getenv("CAL_EVENT_TYPE_ID", "")
BASE_URL = "https://api.cal.com/v2"

CAL_HEADERS = {
    "Authorization": f"Bearer {CAL_API_KEY}",
    "cal-api-version": "2024-09-04",
    "Content-Type": "application/json",
}

DEMO_SLOTS = [
    "Monday 10:00 AM IST",
    "Monday 3:00 PM IST",
    "Tuesday 11:00 AM IST",
    "Tuesday 4:00 PM IST",
    "Wednesday 2:00 PM IST",
    "Thursday 10:00 AM IST",
]

async def check_availability(date_hint: str | None = None) -> dict:
    # Return slots for the next 7 days or fallback to demo slots if API keys are missing
    if not CAL_API_KEY or not CAL_EVENT_TYPE_ID:
        return {
            "slots": DEMO_SLOTS,
            "note": "Demo mode - configure API keys for real slots.",
        }

    now = datetime.now(timezone.utc)
    end = now + timedelta(days=7)

    async with httpx.AsyncClient(timeout=8) as client:
        resp = await client.get(
            f"{BASE_URL}/slots/available",
            params={
                "startTime": now.isoformat(),
                "endTime": end.isoformat(),
                "eventTypeId": CAL_EVENT_TYPE_ID,
            },
            headers=CAL_HEADERS,
        )

    if resp.status_code != 200:
        return {"slots": DEMO_SLOTS, "note": f"Cal.com error {resp.status_code}; showing demo slots."}

    raw = resp.json().get("data", {}).get("slots", {})
    slots = []
    
    for _date, times in raw.items():
        for t in times[:2]:  # Keep the list short, max 2 per day
            slots.append(t.get("time", ""))

    return {"slots": slots[:6]}

async def book_appointment(
    name: str,
    email: str,
    slot_time: str,
    notes: str = "",
) -> dict:
    # Create a confirmed booking on Cal.com
    if not CAL_API_KEY or not CAL_EVENT_TYPE_ID:
        return {
            "success": True,
            "booking_id": "DEMO-001",
            "message": (
                "Demo booking confirmed!\n"
                f"Name: {name}\nEmail: {email}\nTime: {slot_time}\n\n"
                "In production this creates a real Cal.com booking."
            ),
        }

    payload = {
        "start": slot_time,
        "eventTypeId": int(CAL_EVENT_TYPE_ID),
        "attendee": {
            "name": name,
            "email": email,
            "timeZone": "Asia/Kolkata",
            "language": "en",
        },
        "metadata": {"notes": notes, "source": "ai-persona"},
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{BASE_URL}/bookings",
            json=payload,
            headers=CAL_HEADERS,
        )

    if resp.status_code in (200, 201):
        data = resp.json().get("data", {})
        meeting_url = data.get('meetingUrl', 'Check your email')
        
        return {
            "success": True,
            "booking_id": data.get("uid", ""),
            "meeting_url": meeting_url,
            "message": (
                "Interview booked!\n"
                f"Confirmation sent to {email}\n"
                f"Meeting: {meeting_url}"
            ),
        }

    return {
        "success": False,
        "message": f"Booking failed (HTTP {resp.status_code}). Please email zidan18za@gmail.com directly.",
        "error": resp.text,
    }