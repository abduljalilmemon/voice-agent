from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uuid
import json
import base64

from app.agent.graph import agent
from app.voice.tts import text_to_speech_bytes
from app.voice.stt import transcribe_audio_bytes
from app.integrations.booking_store import save_booking, get_all_bookings
from langchain_core.messages import HumanMessage

# load business config once at startup
with open("data/business_config.json") as f:
    BIZ_CONFIG = json.load(f)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Voice Agent API starting — {BIZ_CONFIG['business_name']}...")
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ── Helper ─────────────────────────────────────────────────────

def handle_booking(result: dict, session_id: str) -> dict | None:
    """Check if booking is complete and save it. Returns booking_id or None."""
    booking_data = result.get("booking_data", {})
    if booking_data.get("complete"):
        booking_id = save_booking(
            session_id,
            BIZ_CONFIG["business_name"],
            booking_data
        )
        print(f"  [Main] -> booking #{booking_id} saved for {booking_data.get('name')}")
        return booking_id
    return None

# ── REST Endpoints ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "service": "Voice Agent API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/bookings")
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

@app.post("/chat")
def chat(payload: dict):
    message = payload.get("message", "")
    session_id = payload.get("session_id", str(uuid.uuid4()))

    if not message:
        return {"error": "message is required"}

    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "intent": "",
            "context": "",
            "session_id": session_id,
            "booking_data": {}
        },
        config=config
    )

    reply = result["messages"][-1].content
    booking_id = handle_booking(result, session_id)

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
    await websocket.accept()
    config = {"configurable": {"thread_id": session_id}}
    print(f"WebSocket connected: {session_id[:8]}...")

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message", "")

            if not message:
                continue

            print(f"  WS [{session_id[:8]}] -> {message}")

            await websocket.send_text(json.dumps({
                "type": "typing",
                "session_id": session_id
            }))

            result = agent.invoke(
                {
                    "messages": [HumanMessage(content=message)],
                    "intent": "",
                    "context": "",
                    "session_id": session_id,
                    "booking_data": {}
                },
                config=config
            )

            reply = result["messages"][-1].content
            intent = result.get("intent", "")

            # save booking if complete
            booking_id = handle_booking(result, session_id)

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
    await websocket.accept()
    config = {"configurable": {"thread_id": session_id}}
    print(f"Voice WebSocket connected: {session_id[:8]}...")

    try:
        while True:
            data = await websocket.receive_bytes()
            print(f"  [Voice WS] received {len(data)} bytes of audio")

            # step 1 — transcribe
            await websocket.send_text(json.dumps({"type": "status", "message": "Transcribing..."}))
            transcript = transcribe_audio_bytes(data, mimetype="audio/webm")

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

            # step 2 — run agent
            await websocket.send_text(json.dumps({"type": "status", "message": "Thinking..."}))

            result = agent.invoke(
                {
                    "messages": [HumanMessage(content=transcript)],
                    "intent": "",
                    "context": "",
                    "session_id": session_id,
                    "booking_data": {}
                },
                config=config
            )

            reply = result["messages"][-1].content
            intent = result.get("intent", "")

            # save booking if complete
            booking_id = handle_booking(result, session_id)

            await websocket.send_text(json.dumps({
                "type": "reply",
                "text": reply,
                "intent": intent,
                "booking_confirmed": booking_id is not None,
                "booking_id": booking_id
            }))

            # step 3 — text to speech
            await websocket.send_text(json.dumps({"type": "status", "message": "Generating voice..."}))
            audio_bytes = text_to_speech_bytes(reply)

            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            await websocket.send_text(json.dumps({
                "type": "audio",
                "data": audio_b64,
                "format": "mp3"
            }))

            print(f"  [Voice WS] sent {len(audio_bytes)} bytes of audio")

    except WebSocketDisconnect:
        print(f"Voice WebSocket disconnected: {session_id[:8]}...")