# ThinkAI Voice Agent — Technical Documentation

> **Last updated:** 2026-03-07  
> **Framework:** LiveKit Agents v1.4.4  
> **Status:** Working — local dev tested, production deployment pending

---

## Architecture Overview

```
┌──────────────┐        ┌──────────────────────┐        ┌───────────────┐
│  Browser     │  WS/   │  LiveKit Cloud       │  WS    │  Agent Worker │
│  (widget)    │◄─────► │  (WebRTC relay)      │◄──────►│  (server.py)  │
│              │  WebRTC│  Germany 2 region     │        │               │
└──────┬───────┘        └──────────────────────┘        └───────┬───────┘
       │ HTTP                                                   │
       ▼                                                        │
┌──────────────┐                                                │
│  Web Server  │                                                │
│  (web_server)│                                                │
│  :8000       │                                                │
└──────────────┘                                                │
                                                                ▼
                                              ┌─────────────────────────────┐
                                              │  External Services          │
                                              │  ├─ Google STT              │
                                              │  ├─ Anthropic Claude (LLM)  │
                                              │  ├─ Cartesia TTS            │
                                              │  ├─ Brevo (email)           │
                                              │  ├─ Google Calendar API     │
                                              │  └─ Open-Meteo (weather)    │
                                              └─────────────────────────────┘
```

**Two processes** run in parallel:
1. `server.py` — LiveKit agent worker (connects to LiveKit Cloud via WebSocket)
2. `web_server.py` — FastAPI (serves widget HTML + generates room tokens)

`start.sh` launches both from a single command for production deployment.

---

## File Structure

```
thinkai-voice-agent/
├── server.py              # LiveKit agent worker (STT/LLM/TTS pipeline)
├── web_server.py          # FastAPI web server (widget + /api/token)
├── tools.py               # 6 function tools for the agent
├── voice-widget.html      # Browser-based voice UI (LiveKit JS SDK)
├── start.sh               # Single-command launcher for both processes
├── .env                   # Environment variables (secrets)
├── google-credentials.json # Google service account (not in git)
├── tasks.json             # Local task storage (created at runtime)
├── requirements.txt       # Python dependencies (needs regeneration)
└── AGENT_DOCS.md          # This file
```

---

## Components

### 1. Agent Worker (`server.py`)

**Framework:** LiveKit Agents v1.4.4 (`livekit-agents`)

| Component | Provider | Plugin | Config |
|-----------|----------|--------|--------|
| **STT** | Google Cloud | `livekit.plugins.google` | Languages: `hu-HU`, `en-US` |
| **LLM** | Anthropic | `livekit.plugins.anthropic` | Model: `claude-haiku-4-5-20251001` |
| **TTS** | Cartesia | `livekit.plugins.cartesia` | Model: `sonic-3`, speed: `1.0`, language: `hu` |
| **VAD** | Silero | `livekit.plugins.silero` | ONNX model, see tuning below |

**Agent class:** `ThinkAIAgent(Agent)`
- `instructions`: ~4000 token system prompt (Hungarian, with company knowledge)
- `tools`: 6 function tools from `tools.py`
- `min_endpointing_delay`: 0.5s
- `max_endpointing_delay`: 5.0s
- `on_enter()`: Says greeting on connect

**VAD (Voice Activity Detection) tuning:**

| Parameter | Value | Default | Purpose |
|-----------|-------|---------|---------|
| `activation_threshold` | 0.6 | 0.5 | Speech detection confidence (0.0–1.0). Higher = less sensitive to noise |
| `min_speech_duration` | 0.1s | 0.05s | Debounce: ignores sounds shorter than this |
| `min_silence_duration` | 0.6s | 0.55s | Pause before agent starts responding |

**Cartesia TTS notes:**
- `word_timestamps=False` — not supported for Hungarian on sonic models
- `speed` must be a **float** (0.6–2.0), NOT a string
- Model `sonic-2` is deprecated; use `sonic-3`
- Default voice ID: `36e0c00b-1bfd-4ad7-a0e8-928d4cadca00`

**Startup command:**
```bash
# Dev (hot-reload on file changes):
NO_COLOR=1 python server.py dev

# Production:
python server.py start
```

---

### 2. Web Server (`web_server.py`)

**Framework:** FastAPI + Uvicorn

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serves `voice-widget.html` |
| `/widget` | GET | Serves `voice-widget.html` (alias) |
| `/api/token` | GET | Generates LiveKit room token |
| `/api/health` | GET | Health check (`{"status": "ok"}`) |

**Token generation logic:**
- Creates a unique room: `thinkai-{random_8_hex}`
- Creates a unique participant: `user-{random_6_hex}`
- Grants: `room_join=True` for that specific room
- Returns: `{token, url, room}`

**CORS origins:**
- `https://thinkai.hu` / `https://www.thinkai.hu`
- `http://localhost:3000` / `http://localhost:8000` / `http://127.0.0.1:8000`

**Port:** `PORT` env var (default `8000`)

---

### 3. Tools (`tools.py`)

All tools use `@function_tool` decorator with `Annotated` type hints for parameter descriptions.

| # | Tool | API | Purpose |
|---|------|-----|---------|
| 1 | `send_followup_email` | Brevo SMTP API | Send HTML follow-up email to a prospect |
| 2 | `check_calendar` | Google Calendar API | List events for next N days |
| 3 | `book_meeting` | Google Calendar API | Create calendar event with optional attendee |
| 4 | `get_weather` | Open-Meteo (free) | Current weather for a city (15 cities hardcoded) |
| 5 | `create_task` | Local JSON file | Save a task/note to `tasks.json` |
| 6 | `lookup_info` | In-memory dict | Query internal knowledge base (pricing, audit, tech, team, eaisy, guarantee) |

**Brevo API key handling:**
The `.env` `BREVO_API_KEY` is base64-encoded (MCP format). `tools.py` auto-decodes it:
```python
decoded = base64.b64decode(raw_key).decode()
api_key = json.loads(decoded).get("api_key", raw_key)
```
The actual key format is `xkeysib-...`.

**Google Calendar:**
- Uses service account credentials from `google-credentials.json` or `GOOGLE_APPLICATION_CREDENTIALS_JSON` env var
- Calendar ID from `GOOGLE_CALENDAR_ID` env var (default: `"primary"`)
- Timezone: `Europe/Budapest`

---

### 4. Voice Widget (`voice-widget.html`)

**SDK:** `livekit-client@2` (ESM import from CDN)

**3 visual states:**

| State | Button Color | Status Text | Trigger |
|-------|-------------|-------------|---------|
| 🔵 Listening | Blue gradient | "Hallgatlak..." | Default when connected |
| 🟠 Thinking | Orange gradient | "Gondolkodom..." | After user speech detected |
| 🔴 Speaking | Pink/red gradient | "Beszélek..." | Agent `IsSpeakingChanged` event |

**Error recovery:**
- Auto-reconnect with `maxRetries: 5`, `maxRetryDelay: 5000ms`
- Shows "Újracsatlakozás..." during reconnect
- Error state auto-resets after 4 seconds

**Audio settings (WebRTC):**
- `autoGainControl: true`
- `echoCancellation: true`
- `noiseSuppression: true`

**Transcription display:**
- User messages: 🗣️ (blue)
- Agent messages: 🤖 (pink)
- Both `TranscriptionReceived` and `DataReceived` events handled

---

## Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `LIVEKIT_URL` | ✅ | `wss://thinkai-ugyfelszolgalat-f05w09v7.livekit.cloud` |
| `LIVEKIT_API_KEY` | ✅ | LiveKit Cloud API key |
| `LIVEKIT_API_SECRET` | ✅ | LiveKit Cloud API secret |
| `ANTHROPIC_API_KEY` | ✅ | Anthropic Claude API key (`sk-ant-...`) |
| `CARTESIA_API_KEY` | ✅ | Cartesia TTS API key (`sk_car_...`) |
| `BREVO_API_KEY` | ✅ | Brevo transactional email key (base64-encoded MCP format) |
| `CARTESIA_VOICE_ID` | ❌ | Override default voice (default: `36e0c00b-...`) |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | ❌ | Google SA creds JSON (for cloud deploy) |
| `GOOGLE_CALENDAR_ID` | ❌ | Calendar ID (default: `"primary"`) |
| `PORT` | ❌ | Web server port (default: `8000`) |

---

## System Prompt Summary

~4000 tokens, Hungarian. Key sections:
1. **Personality** — Confident, friendly, professional. Max 2 sentences per reply
2. **Speaking style** — Natural filler words ("Hát...", "Szóval..."), not robotic
3. **Language** — Hungarian default, switches to English if user speaks English
4. **Company info** — Mission, team, 3 pillars, work process, sectors, success stories
5. **Tools** — Lists 6 capabilities and instructs to use them proactively
6. **Email/phone protocol** — Spell-back confirmation, normalization rules for STT artifacts
7. **Conversation memory** — Remember user name, company, and prior context

---

## Known Issues & Gotchas

1. **`requirements.txt` is stale** — Still contains Pipecat/Deepgram from the old framework. Needs regeneration with `pip freeze > requirements.txt`
2. **LiveKit Agents v1.4.4 has no `api_server` option** — Cannot embed FastAPI into the agent worker process. Must run as two separate processes
3. **Cartesia `sonic-2` is deprecated** — Always use `sonic-3`
4. **Hungarian TTS `word_timestamps`** — Not supported, must be `False`
5. **Brevo API key encoding** — The key in `.env` is base64-encoded MCP format; `tools.py` handles decoding
6. **NO_COLOR=1** is needed in dev mode to avoid Rich console output eating logs

---

## Running Locally

```bash
# 1. Activate venv
source /root/ugyfelszolg/thinkai-ugyfelszolg/.venv-linux/bin/activate

# 2. Start both processes (dev mode)
cd thinkai-voice-agent
./start.sh dev

# Or start separately:
NO_COLOR=1 python server.py dev   # agent worker (connects to LiveKit Cloud)
python web_server.py              # web server on :8000

# 3. Open browser
# http://localhost:8000/widget
```

---

## Production Deployment (Railway)

**Two services needed** (or one with `start.sh`):

| Service | Command | Port |
|---------|---------|------|
| Agent Worker | `python server.py start` | None (outbound WS only) |
| Web Server | `python web_server.py` | `$PORT` (Railway assigns) |
| Combined | `./start.sh start` | `$PORT` |

**Required env vars:** All from the table above must be set in Railway.

**Google credentials:** Set `GOOGLE_APPLICATION_CREDENTIALS_JSON` with the full JSON content of the service account file. The code writes it to disk at startup.
