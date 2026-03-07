"""
ThinkAI Voice Agent — Web Server
Serves the voice widget page and generates LiveKit room tokens.
Includes CORS for embedding on thinkai.hu.
"""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from livekit.api import AccessToken, VideoGrants

THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR / ".env")

app = FastAPI(title="ThinkAI Voice Agent")

# CORS — allow embedding on thinkai.hu and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://thinkai.hu",
        "https://www.thinkai.hu",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    return FileResponse(THIS_DIR / "voice-widget.html")


@app.get("/widget")
async def widget():
    return FileResponse(THIS_DIR / "voice-widget.html")


@app.get("/api/token")
async def get_token():
    """Generate a LiveKit room token for a new user."""
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        return JSONResponse({"error": "LiveKit credentials not configured"}, status_code=500)

    room_name = f"thinkai-{uuid.uuid4().hex[:8]}"
    participant_name = f"user-{uuid.uuid4().hex[:6]}"

    token = (
        AccessToken(api_key, api_secret)
        .with_identity(participant_name)
        .with_name("Visitor")
        .with_grants(VideoGrants(
            room_join=True,
            room=room_name,
        ))
    )

    return JSONResponse({
        "token": token.to_jwt(),
        "url": os.getenv("LIVEKIT_URL"),
        "room": room_name,
    })


@app.get("/api/health")
async def health():
    return {"status": "ok", "agent": "thinkai-voice-agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
