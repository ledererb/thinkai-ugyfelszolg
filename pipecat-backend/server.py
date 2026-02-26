import asyncio
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Pipecat backend is running"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-5-sonnet-20241022",
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID", "79a125e8-cd45-4c13-8a67-188112f4dd22"),
    )

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
        }
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

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
        PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        messages.append(
            {
                "role": "system",
                "content": "Köszöntsd a látogatót a nyitó mondatoddal: \"Szia! A ThinkAI asszisztense vagyok. Miben segíthetek?\"",
            }
        )
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
