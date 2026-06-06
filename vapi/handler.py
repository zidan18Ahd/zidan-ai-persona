"""
Vapi webhook handler.
Vapi calls this URL for 3 events:
1. assistant-request: returns the dynamic AI configuration
2. function-call: executes calendar bookings or answers questions
3. end-of-call-report: logs the final call details
"""

import os
from rag.retriever import get_answer
from tools.calendar import check_availability, book_appointment

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

async def handle_webhook(body: dict) -> dict:
    message = body.get("message", {})
    msg_type = message.get("type", "")

    # 1. Assistant Request
    if msg_type == "assistant-request":
        return {"assistant": _build_assistant_config()}

    # 2. Function Call
    if msg_type == "function-call":
        func = message.get("functionCall", {})
        name = func.get("name", "")
        params = func.get("parameters", {})

        if name == "check_availability":
            result = await check_availability(params.get("date"))
            slots = result.get("slots", [])
            if slots:
                slots_str = "\n".join(f"- {s}" for s in slots[:5])
                return {"result": f"Available slots:\n{slots_str}"}
            return {"result": "No slots available this week. Please suggest another week."}

        if name == "book_appointment":
            result = await book_appointment(
                name=params.get("name", ""),
                email=params.get("email", ""),
                slot_time=params.get("slot_time", ""),
                notes=params.get("notes", ""),
            )
            return {"result": result.get("message", "Booking failed.")}

        if name == "answer_question":
            answer = await get_answer(
                params.get("question", ""),
                mode="voice",
            )
            return {"result": answer}

    # 3. End of Call Report
    if msg_type == "end-of-call-report":
        _log_call(message)
        return {"status": "ok"}

    # 4. Status Updates (ignore)
    return {"status": "ok"}

def _log_call(message: dict) -> None:
    duration = message.get("durationSeconds", 0)
    recording = message.get("recordingUrl", "")
    summary = message.get("summary", "")
    print(
        f"Call ended | duration={duration}s | "
        f"recording={recording} | summary={summary[:100]}"
    )

def _build_assistant_config() -> dict:
    # Pull the voice prompt directly from our local config
    from rag.prompts import VOICE_SYSTEM_PROMPT

    return {
        "name": "Zidan's AI Representative",
        "voice": {
            "provider": "11labs",
            "voiceId": ELEVENLABS_VOICE_ID,
            "stability": 0.5,
            "similarityBoost": 0.75,
        },
        "model": {
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": VOICE_SYSTEM_PROMPT}
            ],
            "functions": _get_functions(),
            "temperature": 0.2,
            "maxTokens": 200, 
        },
        "firstMessage": (
            "Hi! I'm Zidan Ahmed's AI representative. "
            "I can tell you about his research and background, or help you schedule an interview. "
            "What would you like to know?"
        ),
        "endCallFunctionEnabled": True,
        "silenceTimeoutSeconds": 30,
        "maxDurationSeconds": 600,
        "backgroundSound": "off",
        "backchannelingEnabled": True,
        "responseDelaySeconds": 0,
    }

def _get_functions() -> list:
    return [
        {
            "name": "check_availability",
            "description": "Check Zidan's available interview slots for the coming week. Call this when the user asks about scheduling, availability, or booking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Optional specific week or date range mentioned by the caller",
                    }
                },
            },
        },
        {
            "name": "book_appointment",
            "description": "Book a confirmed interview slot with Zidan once the caller has chosen a time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Caller's full name"},
                    "email": {"type": "string", "description": "Caller's email for calendar invite"},
                    "slot_time": {"type": "string", "description": "Chosen slot (ISO or natural language)"},
                    "notes": {"type": "string", "description": "Role, company, or any extra context"},
                },
                "required": ["name", "email", "slot_time"],
            },
        },
        {
            "name": "answer_question",
            "description": "Answer detailed questions about Zidan's research, projects, skills, or GitHub repos using RAG over his actual resume and repositories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The caller's question about Zidan's background or work",
                    }
                },
                "required": ["question"],
            },
        },
    ]