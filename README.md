# ThinkAI Voice Agent — Ügyfélszolgálat

Hangalapú AI asszisztens (Pipecat + Deepgram STT + Claude LLM + Cartesia TTS).

---

## Indítás

### 1. Backend (Pipecat szerver)

```powershell
.venv\Scripts\python.exe thinkai-voice-agent\server.py
```

- FastAPI HTTP: `http://localhost:8000`
- Pipecat WebSocket: `ws://localhost:8765`

---

### 2. Frontend (Voice Widget)

Külön terminálban:

```powershell
.venv\Scripts\python.exe -m http.server 3000 --directory thinkai-voice-agent
```

Ezután böngészőben:

```
http://localhost:3000/voice-widget.html
```

> ⚠️ **Fontos:** A widgetet mindig HTTP szerveren keresztül nyisd meg (ne `file://` protokollal), különben a mikrofon és a WebSocket nem fog működni.

---

## Leállítás

Mindkét terminálban: `Ctrl+C`

Ha a port 8765 foglalt marad:

```powershell
$pid = (Get-NetTCPConnection -LocalPort 8765 -State Listen).OwningProcess
Stop-Process -Id $pid -Force
```
