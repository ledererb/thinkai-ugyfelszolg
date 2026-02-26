"""
ThinkAI Voice Agent — Backend Server
FastAPI health check + Pipecat WebSocket voice pipeline
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger

# Load environment variables from .env
load_dotenv()

# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="ThinkAI Voice Agent")


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "thinkai-voice-agent"}


# ── Pipecat Voice Pipeline ──────────────────────────────────────────────────

async def run_voice_pipeline():
    """Start the Pipecat WebSocket voice pipeline on port 8765."""
    from deepgram import LiveOptions

    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.serializers.protobuf import ProtobufFrameSerializer
    from pipecat.frames.frames import LLMMessagesFrame, TextFrame
    from pipecat.services.anthropic.llm import AnthropicLLMService
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.transports.websocket.server import (
        WebsocketServerParams,
        WebsocketServerTransport,
    )

    # ── Transport ────────────────────────────────────────────────────────
    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_out_enabled=True,
            add_wav_header=False,
        ),
        host="0.0.0.0",
        port=8765,
    )

    # ── Deepgram STT ─────────────────────────────────────────────────────
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            model="nova-2",
            language="hu",
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
    )

    # ── Context & Aggregators ────────────────────────────────────────────
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
                "NYITÓ MONDAT (ezt mondd elsőként, amikor a felhasználó csatlakozik):\n"
                "\"Szia! A ThinkAI asszisztense vagyok. Miben segíthetek?\"\n\n"
                "TUDÁSBÁZIS:\n"
                "- ThinkAI szolgáltatások: egyedi AI fejlesztés, pályázati tanácsadás, EAISY termékcsalád\n"
                "- Munkamódszer: Audit → Prezentáció → Kiválasztás → Megvalósítás + pénzvisszafizetési garancia\n"
                "- Kiemelt projektek: Smart Számla Értesítő, Egészségügyi Asszisztens, Marketing Disztribútor\n"
                "- Célszektorok: gyártás, logisztika, pénzügy & jog\n"
                "- Partnerek: BDPST Koncept, Develor, WSZL, ClearService, Duna Autó\n\n"
                "CTA (ha a látogató érdeklődik):\n"
                "- \"Töltsd ki az ajánlatkérő űrlapot a weboldalon!\"\n"
                "- vagy: \"Írj nekünk a hello@thinkai.hu e-mail címre.\""
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
            transport.input(),   # Receive audio from client
            stt,                 # Speech-to-text (Deepgram)
            user_aggregator,     # Add user message to context
            llm,                 # Language model (Claude)
            tts,                 # Text-to-speech (Cartesia)
            transport.output(),  # Send audio back to client
            assistant_aggregator,  # Add bot response to context
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Send initial greeting via TTS
        greeting = "Szia! A ThinkAI asszisztense vagyok. Miben segíthetek?"
        await task.queue_frames(
            [
                LLMMessagesFrame(
                    [{"role": "assistant", "content": greeting}]
                ),
                TextFrame(text=greeting),
            ]
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")

    # ── Run ───────────────────────────────────────────────────────────────
    # Ignore SIGINT in the pipeline task — only uvicorn should handle it.
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    runner = PipelineRunner(handle_sigint=False)
    logger.info("Starting Pipecat voice pipeline on ws://0.0.0.0:8765")
    await runner.run(task)


# ── Startup: launch Pipecat alongside FastAPI ────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Launch the Pipecat pipeline as a background task alongside FastAPI."""
    asyncio.create_task(run_voice_pipeline())
    logger.info("Voice pipeline background task started")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting ThinkAI Voice Agent server...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
