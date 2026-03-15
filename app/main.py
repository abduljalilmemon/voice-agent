from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uuid
import json

from app.agent.graph import agent
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
    """Single turn REST endpoint"""
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
    intent = result.get("intent", "")

    return {
        "session_id": session_id,
        "message": message,
        "reply": reply,
        "intent": intent
    }

# ── WebSocket Endpoint ─────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    config = {"configurable": {"thread_id": session_id}}
    print(f"WebSocket connected: {session_id[:8]}...")

    try:
        while True:
            # receive message from client
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message", "")

            if not message:
                continue

            print(f"  WS [{session_id[:8]}] -> {message}")

            # send typing indicator
            await websocket.send_text(json.dumps({
                "type": "typing",
                "session_id": session_id
            }))

            # run agent
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

            # send response back
            await websocket.send_text(json.dumps({
                "type": "message",
                "session_id": session_id,
                "message": message,
                "reply": reply,
                "intent": intent
            }))

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id[:8]}...")