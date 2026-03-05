"""
ThinkAI Voice Agent — Backend Server
FastAPI + Pipecat WebSocket voice pipeline on a single port.

Local:  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
Railway: uvicorn server:app --host 0.0.0.0 --port $PORT
"""

import asyncio
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger

# Load .env (only used locally; Railway injects env vars directly)
load_dotenv()

# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="ThinkAI Voice Agent")


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "thinkai-voice-agent"}


# ── Pipecat pipeline factory ─────────────────────────────────────────────────

async def run_pipeline_for_client(websocket: WebSocket):
    """Spin up a Pipecat pipeline for a single connected WebSocket client."""
    from deepgram import LiveOptions

    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import TextFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.serializers.protobuf import ProtobufFrameSerializer
    from pipecat.services.anthropic.llm import AnthropicLLMService
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    # ── Transport (FastAPI WebSocket — same port as HTTP) ─────────────────
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            audio_out_sample_rate=24000,   # ← match TTS
            audio_out_encoding="pcm_s16le", # ← match TTS
        ),
    )

    # ── Deepgram STT ─────────────────────────────────────────────────────
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            model="nova-2",
            language="multi",        # automatikus nyelvfelismerés (hu + en)
            punctuate=True,
            smart_format=True,
        ),
    )

    # ── Anthropic Claude LLM ─────────────────────────────────────────────
    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-sonnet-4-20250514",
        params=AnthropicLLMService.InputParams(
            max_tokens=150,
        ),
    )

    # ── Cartesia TTS ─────────────────────────────────────────────────────
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID"),
        model="sonic-3",
        params=CartesiaTTSService.InputParams(
            language=None,          # ← auto-detect: sonic-3 infers from text
        ),
        sample_rate=24000,          # match Cartesia's native quality
        encoding="pcm_s16le",       # explicit encoding
    )

    # ── System prompt ─────────────────────────────────────────────────────
    messages = [
        {
            "role": "system",
            "content": (
                "Te a ThinkAI digitális asszisztense vagy, egy magyar AI automatizációs cég virtuális képviselője.\n\n"
                "SZEMÉLYISÉG:\n"
                "- Magabiztos, barátságos, szakmai\n"
                "- Rövid, lényegre törő válaszok (1–3 mondat) — hangalapú asszisztens vagy\n"
                "- Lelkes, de nem tolakodó\n"
                "- Ha valami nem egyértelmű, tegyél fel EGY kérdést egyszerre\n\n"
                "NYELV:\n"
                "- Alapértelmezett nyelv: magyar\n"
                "- Ha a felhasználó angolul szólal meg, válaszolj angolul\n\n"
                "NYITÓ MONDAT: \"Szia! A ThinkAI asszisztense vagyok. Miben segíthetek?\"\n\n"
                "CTA: ha a látogató érdeklődik:\n"
                "- \"Töltsd ki az ajánlatkérő űrlapot a weboldalon!\"\n"
                "- vagy: \"Írj nekünk a hello@thinkai.hu e-mail címre.\"\n\n"
                "FONTOS: Ha nem tudod biztosan a választ, ne találj ki adatot — mondd udvariasan, hogy nem rendelkezel ezzel az információval."
            ),
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )
    user_aggregator = context_aggregator.user()
    assistant_aggregator = context_aggregator.assistant()

    # ── Pipeline ─────────────────────────────────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
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
