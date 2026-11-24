from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import uuid
import json
import base64
from pydantic import BaseModel, Field

from app.agent.graph import agent
from app.voice.tts import text_to_speech_bytes
from app.voice.stt import transcribe_audio_bytes
from app.integrations.booking_store import get_all_bookings
from app.config import BIZ_CONFIG
from app.security import (
    ALLOWED_ORIGINS, MAX_AUDIO_BYTES, MAX_MESSAGE_CHARS, MAX_SESSION_ID_CHARS,
    client_key, require_admin, turn_limiter, websocket_origin_allowed,
)
from langchain_core.messages import HumanMessage

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Voice Agent API starting -- {BIZ_CONFIG['business_name']}...")
    yield
    print("Voice Agent API shutting down...")

app = FastAPI(
    title="Voice Agent API",
    description="AI Voice Agent for Business Customer Support",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # empty by default: the bundled frontend is same-origin
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ── Helper ─────────────────────────────────────────────────────

def run_turn(session_id: str, text: str) -> dict:
    """Run one conversation turn through the agent (blocking -- call via asyncio.to_thread).

    Only per-turn fields are passed in. Persistent fields (booking_data, booking_id)
    are left out so the checkpointer keeps them between turns.
    """
    return agent.invoke(
        {
            "messages": [HumanMessage(content=text)],
            "intent": "",
            "context": "",
            "session_id": session_id,
            "new_booking_id": None
        },
        config={"configurable": {"thread_id": session_id}}
    )

RATE_LIMITED = "You're sending messages too quickly. Please wait a moment and try again."

async def accept_if_allowed(websocket: WebSocket, session_id: str) -> bool:
    """Reject cross-site browser connections and malformed session ids before accepting."""
    if not websocket_origin_allowed(websocket) or len(session_id) > MAX_SESSION_ID_CHARS:
        await websocket.close(code=1008)  # policy violation
        return False
    await websocket.accept()
    return True

async def send_error(websocket: WebSocket, message: str):
    await websocket.send_text(json.dumps({"type": "error", "message": message}))

# ── REST Endpoints ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "service": "Voice Agent API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/bookings", dependencies=[Depends(require_admin)])
def list_bookings():
    rows = get_all_bookings()
    bookings = []
    for row in rows:
        bookings.append({
            "id": row[0],
            "session_id": row[1],
            "business": row[2],
            "name": row[3],
            "phone": row[4],
            "email": row[5],
            "date": row[6],
            "time": row[7],
            "service": row[8],
            "status": row[10],
            "created_at": row[11]
        })
    return {"total": len(bookings), "bookings": bookings}

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=MAX_SESSION_ID_CHARS)

@app.post("/chat")
def chat(payload: ChatRequest, request: Request):
    if not turn_limiter.allow(client_key(request.client.host if request.client else None)):
        raise HTTPException(status_code=429, detail=RATE_LIMITED)

    message, session_id = payload.message, payload.session_id
    result = run_turn(session_id, message)

    reply = result["messages"][-1].content
    booking_id = result.get("new_booking_id")

    return {
        "session_id": session_id,
        "message": message,
        "reply": reply,
        "intent": result.get("intent", ""),
        "booking_saved": booking_id is not None
    }

# ── Text WebSocket ─────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    if not await accept_if_allowed(websocket, session_id):
        return
    print(f"WebSocket connected: {session_id[:8]}...")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = str(json.loads(data).get("message", "")).strip()
            except (ValueError, AttributeError):
                await send_error(websocket, 'Expected JSON like {"message": "..."}')
                continue

            if not message:
                continue
            if len(message) > MAX_MESSAGE_CHARS:
                await send_error(websocket, f"Message too long (max {MAX_MESSAGE_CHARS} characters).")
                continue
            if not turn_limiter.allow(client_key(websocket.client.host if websocket.client else None)):
                await send_error(websocket, RATE_LIMITED)
                continue

            print(f"  WS [{session_id[:8]}] -> {message}")

            await websocket.send_text(json.dumps({
                "type": "typing",
                "session_id": session_id
            }))

            result = await asyncio.to_thread(run_turn, session_id, message)

            reply = result["messages"][-1].content
            intent = result.get("intent", "")

            # set only on the turn the booking was saved
            booking_id = result.get("new_booking_id")

            await websocket.send_text(json.dumps({
                "type": "message",
                "session_id": session_id,
                "message": message,
                "reply": reply,
                "intent": intent,
                "booking_confirmed": booking_id is not None,
                "booking_id": booking_id
            }))

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id[:8]}...")

# ── Voice WebSocket ────────────────────────────────────────────

@app.websocket("/ws/voice/{session_id}")
async def voice_websocket_endpoint(websocket: WebSocket, session_id: str):
    if not await accept_if_allowed(websocket, session_id):
        return
    print(f"Voice WebSocket connected: {session_id[:8]}...")

    try:
        while True:
            data = await websocket.receive_bytes()
            print(f"  [Voice WS] received {len(data)} bytes of audio")

            if len(data) > MAX_AUDIO_BYTES:
                await send_error(websocket, "Recording too long. Please keep it under a few minutes.")
                continue
            if not turn_limiter.allow(client_key(websocket.client.host if websocket.client else None)):
                await send_error(websocket, RATE_LIMITED)
                continue

            # step 1 -- transcribe
            await websocket.send_text(json.dumps({"type": "status", "message": "Transcribing..."}))
            transcript = await asyncio.to_thread(transcribe_audio_bytes, data, "audio/webm")

            if not transcript.strip():
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Could not transcribe audio. Please try again."
                }))
                continue

            print(f"  [Voice WS] transcript: '{transcript}'")

            await websocket.send_text(json.dumps({
                "type": "transcript",
                "text": transcript
            }))

            # step 2 -- run agent
            await websocket.send_text(json.dumps({"type": "status", "message": "Thinking..."}))

            result = await asyncio.to_thread(run_turn, session_id, transcript)

            reply = result["messages"][-1].content
            intent = result.get("intent", "")

            # set only on the turn the booking was saved
            booking_id = result.get("new_booking_id")

            await websocket.send_text(json.dumps({
                "type": "reply",
                "text": reply,
                "intent": intent,
                "booking_confirmed": booking_id is not None,
                "booking_id": booking_id
            }))

            # step 3 -- text to speech
            await websocket.send_text(json.dumps({"type": "status", "message": "Generating voice..."}))
            audio_bytes = await text_to_speech_bytes(reply)

            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            await websocket.send_text(json.dumps({
                "type": "audio",
                "data": audio_b64,
                "format": "mp3"
            }))

            print(f"  [Voice WS] sent {len(audio_bytes)} bytes of audio")

    except WebSocketDisconnect:
        print(f"Voice WebSocket disconnected: {session_id[:8]}...")