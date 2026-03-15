from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import uuid
import json
import base64
import asyncio

from app.agent.graph import agent
from app.voice.tts import text_to_speech_bytes
from app.voice.stt import transcribe_audio_bytes
from langchain_core.messages import HumanMessage

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Voice Agent API starting...")
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

# ── REST Endpoints ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "service": "Voice Agent API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

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
            "session_id": session_id
        },
        config=config
    )

    reply = result["messages"][-1].content
    return {
        "session_id": session_id,
        "message": message,
        "reply": reply,
        "intent": result.get("intent", "")
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
                    "session_id": session_id
                },
                config=config
            )

            reply = result["messages"][-1].content
            intent = result.get("intent", "")

            await websocket.send_text(json.dumps({
                "type": "message",
                "session_id": session_id,
                "message": message,
                "reply": reply,
                "intent": intent
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
            # receive audio bytes from browser
            data = await websocket.receive_bytes()
            print(f"  [Voice WS] received {len(data)} bytes of audio")

            # step 1 — transcribe audio to text
            await websocket.send_text(json.dumps({"type": "status", "message": "Transcribing..."}))
            transcript = transcribe_audio_bytes(data, mimetype="audio/webm")

            if not transcript.strip():
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Could not transcribe audio. Please try again."
                }))
                continue

            print(f"  [Voice WS] transcript: '{transcript}'")

            # send transcript back so UI can show it
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
                    "session_id": session_id
                },
                config=config
            )

            reply = result["messages"][-1].content
            intent = result.get("intent", "")
            print(f"  [Voice WS] reply: '{reply[:60]}...'")

            # send text reply
            await websocket.send_text(json.dumps({
                "type": "reply",
                "text": reply,
                "intent": intent
            }))

            # step 3 — convert reply to speech
            await websocket.send_text(json.dumps({"type": "status", "message": "Generating voice..."}))
            audio_bytes = text_to_speech_bytes(reply)

            # send audio as base64 so browser can play it
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            await websocket.send_text(json.dumps({
                "type": "audio",
                "data": audio_b64,
                "format": "mp3"
            }))

            print(f"  [Voice WS] sent {len(audio_bytes)} bytes of audio")

    except WebSocketDisconnect:
        print(f"Voice WebSocket disconnected: {session_id[:8]}...")