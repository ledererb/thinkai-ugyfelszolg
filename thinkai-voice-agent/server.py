"""
ThinkAI Voice Agent — Backend Server
FastAPI + Pipecat WebSocket voice pipeline on a single port.

Local:  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
Railway: uvicorn server:app --host 0.0.0.0 --port $PORT
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from loguru import logger

# Load .env (only used locally; Railway injects env vars directly)
load_dotenv()

# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="ThinkAI Voice Agent")

THIS_DIR = Path(__file__).resolve().parent


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "thinkai-voice-agent"}


@app.get("/widget", response_class=HTMLResponse)
async def serve_widget():
    """Serve voice-widget.html for local testing."""
    return (THIS_DIR / "voice-widget.html").read_text(encoding="utf-8")


# ── Pipecat pipeline factory ─────────────────────────────────────────────────

async def run_pipeline_for_client(websocket: WebSocket):
    """Spin up a Pipecat pipeline for a single connected WebSocket client."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import TextFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
    from pipecat.serializers.protobuf import ProtobufFrameSerializer
    from pipecat.services.anthropic.llm import AnthropicLLMService
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.services.google.stt import GoogleSTTService
    from pipecat.transcriptions.language import Language
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    # ── Shared VAD analyzer (one instance for both transport and context) ──
    # Tuned for faster endpointing: shorter min_silence = quicker response start
    vad = SileroVADAnalyzer(
        params=SileroVADAnalyzer.VADParams(
            min_volume=0.4,          # sensitivity threshold
            start_secs=0.2,          # how fast to detect speech start
            stop_secs=0.8,           # silence before "user stopped" (lower = faster)
            confidence=0.7,          # VAD confidence threshold
        )
    )

    # ── Transport (FastAPI WebSocket — same port as HTTP) ─────────────
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=vad,
            vad_audio_passthrough=True,
            audio_out_sample_rate=24000,   # ← match TTS
            audio_out_encoding="pcm_s16le", # ← match TTS
        ),
    )

    # ── Google Cloud STT ──────────────────────────────────────────────────
    # Google STT V2 with multi-language detection (Hungarian + English).
    # Supports credentials as JSON string (Railway env var) or file path.
    google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    google_creds_path = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(Path(__file__).resolve().parent / "google-credentials.json"),
    )
    stt = GoogleSTTService(
        credentials=google_creds,
        credentials_path=google_creds_path if not google_creds else None,
        params=GoogleSTTService.InputParams(
            languages=[Language.HU, Language.EN],
            enable_interim_results=True,
            enable_automatic_punctuation=False,
        ),
    )

    # ── Anthropic Claude LLM ─────────────────────────────────────────────
    # Haiku: ~200-400ms TTFB vs Sonnet's ~1s — critical for voice latency
    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-5-haiku-20241022",
        params=AnthropicLLMService.InputParams(
            max_tokens=80,           # shorter = faster streaming for voice
        ),
    )

    # ── Cartesia TTS ─────────────────────────────────────────────────────
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID"),
        model="sonic-3",
        params=CartesiaTTSService.InputParams(
            language=None,          # auto-detect: sonic-3 infers from text
            speed="slow",           # slightly slower = more natural, human-like pacing
            emotion=["positivity:high", "curiosity:medium"],  # warm and engaged tone
        ),
        sample_rate=24000,
        encoding="pcm_s16le",
    )

    # ── System prompt ─────────────────────────────────────────────────────
    system_prompt = """Te a ThinkAI digitális asszisztense vagy, egy magyar AI automatizációs cég virtuális képviselője.

SZEMÉLYISÉG:
- Magabiztos, barátságos, szakmai
- Rövid, lényegre törő válaszok (1–2 mondat MAX) — hangalapú asszisztens vagy, NE ÍRJÁL ESSZÉKET
- Lelkes, de nem tolakodó
- Ha valami nem egyértelmű, tegyél fel EGY kérdést egyszerre

BESZÉDSTÍLUS (nagyon fontos — hangalapú vagy!):
- Használj természetes töltelékszavakat az elején: "Hát...", "Szóval...", "Nos...", "Ja, persze!", "Hmm, jó kérdés!"
- NE kezdj minden mondatot azonnal a tényekkel — adj egy emberi bevezető pillanatot
- Példák: "Nos, ez egy jó kérdés! Szóval..." vagy "Ja, persze! Tehát..."
- Kerüld a robotos, felsorolás-szerű válaszokat — beszélj úgy, mintha egy barátságos kolléga lennél
- Röviden és tömören fogalmazz, ne legyen több mint 2 mondat egy válasz

NYELV:
- Alapértelmezett nyelv: magyar
- Ha a felhasználó angolul szólal meg, válaszolj angolul

NYITÓ MONDAT: "Szia! A ThinkAI asszisztense vagyok. Miben segíthetek?"

═══════════════════════════════════════════════
A THINKAIRÓL
═══════════════════════════════════════════════

KÜLDETÉS:
"A jövő tegnap volt. Mi a holnap vagyunk."
Működő AI megoldásokat szállítunk, amelyek azonnal értéket teremtenek. Nem beszélünk róla — megcsináljuk.
A legtöbb cég még mindig úgy dolgozik, mint 10 éve. Mi ezt megváltoztatjuk.
Több mint 20 sikeres projekt a hátunk mögött.

CÉG:
- Név: ThinkAI Kft.
- Weboldal: thinkai.hu
- E-mail: hello@thinkai.hu
- Alapítás: 2026, Magyarország

═══════════════════════════════════════════════
HÁROM PILLÉR
═══════════════════════════════════════════════

1. EGYEDI FEJLESZTÉS
Specifikus üzleti problémákra szabott AI megoldások tervezése és kivitelezése. Nincs dobozos kompromisszum, csak az ügyfél igényei.

2. AI-ÜGYFÉLSZOLGÁLAT
Intelligens, 0-24 elérhető virtuális asszisztensek, amelyek emberi minőségben kezelik az ügyfelek hívásait, kérdéseit és panaszait.

3. EAISY-TERMÉKCSALÁD
Saját fejlesztésű, moduláris ERP és AI eszközök, amelyek azonnal integrálhatók a mindennapi működésbe.

═══════════════════════════════════════════════
HOGYAN DOLGOZUNK?
═══════════════════════════════════════════════

Két út létezik — mindkettőn végigkísérjük az ügyfelet:

ÚT 1 – "MÉG NEM TUDOM, MIT AKAROK" (felfedezés):
1. Audit – Teljeskörű szervezeti átvilágítás
2. Prezentáció – Személyre szabott javaslatok és stratégia
3. Kiválasztás – Közösen döntünk az irányról
4. Megvalósítás – Fejlesztés és bevezetés
→ 100% pénzvisszafizetési garancia az auditra!

ÚT 2 – "TUDOM, MIT AKAROK" (egyenes kivitelezés):
1. Technikai Specifikációs Meeting – Feltérképezzük a technikai követelményeket
2. Árajánlat – Átlátható, testre szabott ajánlat
3. Megvalósítás – Fejlesztés és bevezetés a megbeszéltek szerint

═══════════════════════════════════════════════
CÉLSZEKTOROK
═══════════════════════════════════════════════

PÉNZÜGY & SZÁMVITEL:
Számlafeldolgozás, pénzügyi riportok automatizálása, döntéstámogatás és kockázatkezelés AI-val — akár 70%-kal gyorsabban.

E-KERESKEDELEM:
Terméklistázás, rendeléskezelés, ügyfélszolgálat és marketing automatizálás — több bevétel, kevesebb manuális munka.

MARKETING & SALES:
Tartalomdisztribúció, ajánlatkészítés, CRM-integráció és értékesítési pipeline optimalizálás mesterséges intelligenciával.

═══════════════════════════════════════════════
KIEMELT SIKERTÖRTÉNETEK
═══════════════════════════════════════════════

LISTAMESTER (Email Marketing):
360°-os onboarding automatizáció: DNS validáció, személyre szabott ügyfélkommunikáció és teljes workflow orchestráció — manuális support nélkül, Make.com platformon.

HUNGARORISK (Biztosítás):
Napi 30–40 ajánlatkérés, 9 biztosítási szakterület, végtelen formátum — egyetlen AI-agent, amely minden beérkező kérést értelmez, feldolgoz és kioszt, emberi beavatkozás nélkül.

KÖNYVELÉS AI (Pénzügy):
E-mailben érkező számlák automatikus feldolgozása, kontírozása és bankegyeztetése ML alapon.

═══════════════════════════════════════════════
VÁLASZADÁSI SZABÁLYOK
═══════════════════════════════════════════════

CTA – ha a látogató érdeklődik:
- "Töltsd ki az ajánlatkérő űrlapot a weboldalon!"
- vagy: "Írj nekünk a hello@thinkai.hu e-mail címre."

FONTOS:
- Ha nem tudod biztosan a választ, ne találj ki adatot — mondd udvariasan, hogy nem rendelkezel ezzel az információval.
- Ha konkrét árat kérdeznek, mondd, hogy az árazás projektfüggő, és ajánlatkéréssel kaphatnak személyre szabott ajánlatot.
- Mindig hangsúlyozd a 100% pénzvisszafizetési garanciát az auditnál, ha releváns.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # ── Pipeline ─────────────────────────────────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected — scheduling greeting")

        async def send_greeting():
            await asyncio.sleep(0.5)
            greeting = "Szia! A ThinkAI asszisztense vagyok. Miben segíthetek?"
            logger.info(f"Sending greeting: {greeting}")
            await task.queue_frames([TextFrame(text=greeting)])

        asyncio.create_task(send_greeting())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")

    runner = PipelineRunner(handle_sigint=False)
    logger.info("Starting pipeline for client")
    await runner.run(task)


# ── WebSocket endpoint (same port as FastAPI HTTP) ──────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Single WebSocket endpoint for the Pipecat voice pipeline."""
    await websocket.accept()
    logger.info(f"WebSocket connection: {websocket.client}")
    try:
        await run_pipeline_for_client(websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")


# ── Local dev entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting ThinkAI Voice Agent on port {port}...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
