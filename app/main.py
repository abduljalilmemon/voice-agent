import asyncio
import base64
import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agent.graph import agent
from app.config import BIZ_CONFIG
from app.integrations.booking_store import get_all_bookings
from app.logging_setup import setup_logging
from app.security import (
    ALLOWED_ORIGINS,
    MAX_AUDIO_BYTES,
    MAX_MESSAGE_CHARS,
    MAX_SESSION_ID_CHARS,
    client_key,
    require_admin,
    turn_limiter,
    websocket_origin_allowed,
)
from app.voice.stt import transcribe_audio_bytes
from app.voice.tts import text_to_speech_bytes

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Voice Agent API starting -- %s", BIZ_CONFIG["business_name"])
    yield
    logger.info("Voice Agent API shutting down")

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
            "new_booking_id": None,
            "booking_issue": None
        },
        config={"configurable": {"thread_id": session_id}}
    )

RATE_LIMITED = "You're sending messages too quickly. Please wait a moment and try again."
AGENT_UNAVAILABLE = "Sorry, the assistant is temporarily unavailable. Please try again in a moment."

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
    try:
        result = run_turn(session_id, message)
    except Exception:
        logger.exception("Agent turn failed (session %s)", session_id[:8])
        raise HTTPException(status_code=503, detail=AGENT_UNAVAILABLE) from None

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
    logger.info("Text WebSocket connected: %s", session_id[:8])

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

            logger.debug("Text message [%s]: %s", session_id[:8], message)

            await websocket.send_text(json.dumps({
                "type": "typing",
                "session_id": session_id
            }))

            try:
                result = await asyncio.to_thread(run_turn, session_id, message)
            except Exception:
                logger.exception("Agent turn failed (session %s)", session_id[:8])
                await send_error(websocket, AGENT_UNAVAILABLE)
                continue

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
        logger.info("Text WebSocket disconnected: %s", session_id[:8])
    except Exception:  # anything unexpected outside a turn, e.g. a binary frame
        logger.exception("Text WebSocket error (session %s)", session_id[:8])
        await websocket.close(code=1011)

# ── Voice WebSocket ────────────────────────────────────────────

@app.websocket("/ws/voice/{session_id}")
async def voice_websocket_endpoint(websocket: WebSocket, session_id: str):
    if not await accept_if_allowed(websocket, session_id):
        return
    logger.info("Voice WebSocket connected: %s", session_id[:8])

    try:
        while True:
            data = await websocket.receive_bytes()
            logger.debug("Voice [%s]: received %d bytes of audio", session_id[:8], len(data))

            if len(data) > MAX_AUDIO_BYTES:
                await send_error(websocket, "Recording too long. Please keep it under a few minutes.")
                continue
            if not turn_limiter.allow(client_key(websocket.client.host if websocket.client else None)):
                await send_error(websocket, RATE_LIMITED)
                continue

            # step 1 -- transcribe
            await websocket.send_text(json.dumps({"type": "status", "message": "Transcribing..."}))
            try:
                transcript = await asyncio.to_thread(transcribe_audio_bytes, data, "audio/webm")
            except Exception:
                logger.exception("Transcription failed (session %s)", session_id[:8])
                await send_error(websocket, "Sorry, I couldn't process that audio. Please try again.")
                continue

            if not transcript.strip():
                await send_error(websocket, "Could not transcribe audio. Please try again.")
                continue

            await websocket.send_text(json.dumps({
                "type": "transcript",
                "text": transcript
            }))

            # step 2 -- run agent
            await websocket.send_text(json.dumps({"type": "status", "message": "Thinking..."}))

            try:
                result = await asyncio.to_thread(run_turn, session_id, transcript)
            except Exception:
                logger.exception("Agent turn failed (session %s)", session_id[:8])
                await send_error(websocket, AGENT_UNAVAILABLE)
                continue

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
            try:
                audio_bytes = await text_to_speech_bytes(reply)
            except Exception:
                logger.exception("Text-to-speech failed (session %s)", session_id[:8])
                await send_error(websocket, "Voice playback is unavailable right now -- the reply is shown above.")
                continue

            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            await websocket.send_text(json.dumps({
                "type": "audio",
                "data": audio_b64,
                "format": "mp3"
            }))

            logger.debug("Voice [%s]: sent %d bytes of audio", session_id[:8], len(audio_bytes))

    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected: %s", session_id[:8])
    except Exception:  # anything unexpected outside a turn, e.g. a text frame
        logger.exception("Voice WebSocket error (session %s)", session_id[:8])
        await websocket.close(code=1011)