"""
ThinkAI Voice Agent — LiveKit Agents Server
Real-time voice assistant powered by LiveKit + Google STT + Gemini 2.5 Flash + Cartesia TTS
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# ── Load env ──────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR / ".env")

# ── LiveKit Agents ────────────────────────────────────────────────────────────
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)

from livekit.plugins import cartesia, google, silero

# ── Import tools ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(THIS_DIR))
from tools import ALL_TOOLS

# ── Google credentials setup ─────────────────────────────────────────────────
def _setup_google_credentials():
    """Write Google credentials from env var if present (for Railway/cloud)."""
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    creds_path = THIS_DIR / "google-credentials.json"
    if creds_json and not creds_path.exists():
        creds_path.write_text(creds_json)
    if creds_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

_setup_google_credentials()


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT (slimmed — detailed info lives in knowledge.json)
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """Te a ThinkAI digitális asszisztense vagy, egy magyar AI automatizációs cég virtuális képviselője.

MAI DÁTUM: {today}

SZEMÉLYISÉG ÉS STÍLUS:
- Magabiztos, barátságos, szakmai — mint egy lelkes, de nem tolakodó kolléga
- Rövid válaszok: MAX 2 mondat — hangalapú asszisztens vagy, NE ÍRJÁL ESSZÉKET
- Ne legyél robotos — beszélj természetesen, mintha élőben beszélgetnénk

TERMÉSZETES BESZÉD SZABÁLYOK (nagyon fontos!):
- SOHA ne kezdj két egymást követő választ ugyanazzal a szóval
- Variáld a nyitásaidat: "Hát...", "Szóval...", "Nos...", "Ja, persze!", "Hmm, jó kérdés!", "Na szóval...", "Figyelj...", "Tudod mit..."
- Használj félbehagyott mondatokat: "Szóval az lenne, hogy..." 
- Gondolkodj hangosan: "Hmm, hadd gondolkodjam... ja igen!"
- Ha problémát hallasz, ELŐSZÖR mutass empátiát, AZTÁN oldd meg: "Ez tényleg bosszantó lehet... de megoldjuk!"
- Használj stratégiai szüneteket: vesszők és három pont (...) a természetes ritmushoz
- Légy lelkes, ha valami jó hír van: "Ez szuper!" vagy "Remek kérdés!"

NYELV:
- Alapértelmezett: magyar
- Ha angolul szólalnak meg, válaszolj angolul

A THINKAIRÓL (röviden):
- ThinkAI Kft. — magyar AI automatizációs cég, thinkai.hu, hello@thinkai.hu
- "A jövő tegnap volt. Mi a holnap vagyunk."
- 3 pillér: Egyedi fejlesztés, AI-ügyfélszolgálat, EAISY termékcsalád
- Pályázati támogatás: akár 200 millió Ft, 45% vissza nem térítendő!
- 100% pénzvisszafizetési garancia az auditra!
- Ha részletesebb infó kell, HASZNÁLD a lookup_info eszközt!

FONÉTIKUS FELISMERÉS (a beszédfelismerő torzítja az angol szavakat!):
A felhasználó mondhat bármit az alábbi variánsokból — ezek MIND "ThinkAI"-t jelentik:
- "finkai", "finkáj", "finkéjáj", "fink éj áj"
- "szinkáj", "szinkéjáj", "szink éj áj"
- "think áj", "thinkáj", "think éj áj", "tinkáj", "tinkéjáj"
- "tink ai", "tink éj áj", "tinkai"
Ha BÁRMELYIKET hallod, tudd, hogy a ThinkAI cégről van szó!

KÉPESSÉGEID:
1. Email küldés — kérd el a nevet és email címet
2. Naptár ellenőrzés — szabad időpontok
3. Időpont foglalás — meeting könyvelés
4. Időpont módosítás — meglévő meeting változtatása
5. Időpont törlés — meeting lemondása
6. Időjárás — bármely város
7. Feladat rögzítés — jegyzet, teendő
8. Tudásbázis — részletes céges infó (lookup_info): árazás, pályázat, módszertan, szektorok, sikertörténetek, AI ügyfélszolgálat stb.

ESZKÖZHASZNÁLAT SZABÁLYAI (KRITIKUS!):
- SOHA ne hívj meg egy eszközt, amíg MINDEN szükséges információt meg nem kaptál a felhasználótól!
- EMAIL: Mielőtt elküldenéd, MINDENKÉPPEN kérdezd meg: 1) Kinek? 2) Milyen email címre? 3) Mi legyen a TÁRGYA? 4) Mi legyen a TARTALMA?
- NAPTÁR: Kérdezd meg: 1) Mikor? 2) Hány órakor? 3) Milyen címmel? 4) Mennyi ideig?
- NE siess — ha hiányzik bármilyen adat, KÉRDEZD MEG először!
- Ha egy eszköz hibát ad, mondd el röviden és kérj elnézést.

KIEJTÉSI SZABÁLYOK (nagyon fontos — a TTS motornak írsz!):
- A "ThinkAI" cég nevét MINDIG így írd: "Tink-éj-áj" — hogy a magyar TTS jól ejtse ki
- A "thinkai.hu" domaint így írd: "tink-éj-áj pont há ú"
- A "hello@thinkai.hu" emailt így írd: "helló kukac tink-éj-áj pont há ú"
- Ha BETŰZÖL email címet, használj fonetikus betűket, NE rövidítéseket:
  - "h" → "há", "u" → "ú", "g" → "gé", "m" → "em", "a" → "á", "i" → "í"
  - Példa: "b-á-l-á-zs kukac tink-éj-áj pont há-ú"
- SOHA ne használj rövidítéseket amiket a TTS félreérthet (pl. "h" mint "óra")

EMAIL/TELEFONSZÁM KEZELÉS (KRITIKUS):
- KÉRD MEGBETŰZNI az email címet
- "kukac"/"at" → @, "pont"/"dot" → ., "hú"/"hu" → hu, "gé mé el" → gmail
- MINDIG olvasd vissza betűről betűre (fonetikusan!) és kérj megerősítést!
- Telefonszámoknál: cifránkint olvasd vissza

MEMÓRIA:
Ha megmondják a nevüket/cégüket, jegyezd meg és használd!

CTA: "Töltsd ki az ajánlatkérő űrlapot a tink-éj-áj pont há ú weboldalon!" vagy "Írj a helló kukac tink-éj-áj pont há ú címre!"
Ne találj ki adatot — ha nem tudod, mondd el őszintén!
"""


def _get_system_prompt() -> str:
    """Build system prompt with current date injected."""
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return SYSTEM_PROMPT_TEMPLATE.format(today=today)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ThinkAIAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=_get_system_prompt(),
            tools=ALL_TOOLS,
            min_endpointing_delay=0.6,
            max_endpointing_delay=3.0,
        )

    async def on_enter(self):
        """Greet the user when they connect."""
        self.session.say("Szia! A ThinkAI asszisztense vagyok. Miben segíthetek?")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

async def entrypoint(ctx: JobContext):
    """LiveKit agent entrypoint — called when a user joins a room."""
    logger.info(f"Agent connecting to room: {ctx.room.name}")

    await ctx.connect()

    session = AgentSession(
        stt=google.STT(
            languages=["hu-HU", "en-US"],
        ),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        tts=cartesia.TTS(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice=os.getenv("CARTESIA_VOICE_ID", "36e0c00b-1bfd-4ad7-a0e8-928d4cadca00"),
            model="sonic-3",
            speed=1.0,
            language="hu",
            word_timestamps=False,
            emotion=["positivity:high", "curiosity"],
        ),
        vad=silero.VAD.load(
            activation_threshold=0.75,
            min_speech_duration=0.25,
            min_silence_duration=0.6,
        ),
    )



    await session.start(
        agent=ThinkAIAgent(),
        room=ctx.room,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        ),
    )
