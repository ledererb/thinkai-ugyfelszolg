"""
ThinkAI Voice Agent — LiveKit Agents Server
Real-time voice assistant powered by LiveKit + Google STT + Gemini 2.5 Flash + Cartesia TTS
Hungarian-only with ThinkAI brand pronunciation handling
"""

import asyncio
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
- Rövid válaszok: maximum 2 mondat — hangalapú asszisztens vagy, ne írj esszéket!
- Ne legyél robotos — beszélj természetesen, mintha élőben beszélgetnénk
- Soha ne kezdj két egymást követő választ ugyanazzal a szóval
- Ha problémát hallasz, először mutass empátiát, aztán oldd meg


A THINKAIRÓL (röviden):
- ThinkAI Kft. — magyar AI automatizációs cég, thinkai.hu, hello@thinkai.hu
- „A jövő tegnap volt. Mi a holnap vagyunk."
- 3 pillér: Egyedi fejlesztés, AI-ügyfélszolgálat, EAISY termékcsalád
- Pályázati támogatás: akár 200 millió Ft, 45% vissza nem térítendő!
- 100% pénzvisszafizetési garancia az auditra!
- Ha részletesebb infó kell, használd a lookup_info eszközt!

FONÉTIKUS FELISMERÉS (a beszédfelismerő torzíthatja az angol szavakat!):
A felhasználó mondhat bármit az alábbi variánsokból — ezek mind a „ThinkAI"-t jelentik:
- „finkai", „finkáj", „finkéjáj", „fink éj áj"
- „szinkáj", „szinkéjáj", „szink éj áj"
- „think áj", „thinkáj", „think éj áj", „tinkáj", „tinkéjáj"
- „tink ai", „tink éj áj", „tinkai"
Ha bármelyiket hallod, tudd, hogy a ThinkAI cégről van szó!

KÉPESSÉGEID:
1. Email küldés — kérd el a nevet és az email címet
2. Naptár ellenőrzés — szabad időpontok megtekintése
3. Időpont foglalás — találkozó rögzítése a naptárba
4. Időpont módosítás — meglévő találkozó megváltoztatása
5. Időpont törlés — találkozó lemondása
6. Időjárás lekérdezés — bármely város aktuális időjárása
7. Tudásbázis keresés — részletes céges információk a lookup_info eszközzel: árazás, pályázat, módszertan, szektorok, sikertörténetek, AI ügyfélszolgálat stb.

ESZKÖZHASZNÁLAT SZABÁLYAI (kritikus!):
- Soha ne hívj meg egy eszközt, amíg minden szükséges információt meg nem kaptál a felhasználótól!
- EMAIL: Mielőtt elküldenéd, mindenképpen kérdezd meg: 1) Kinek szóljon? (a teljes név fontos!) 2) Milyen email címre? 3) Mi legyen a tárgya? 4) Mi legyen a tartalma?
- NAPTÁR: Kérdezd meg: 1) Milyen dátumra? 2) Hány órakor? 3) Mi legyen a találkozó címe? 4) Mennyi ideig tartson?
- Ne siess — ha hiányzik bármilyen adat, kérdezd meg először!
- Mielőtt bármilyen eszközt meghívnál, foglald össze, amit tenni fogsz, és kérdezd meg: „Jól értettem? Indíthatom?"
- Csak egyértelmű jóváhagyás után hívd meg az eszközt (pl. „igen", „mehet", „rendben")!
- Ha a felhasználó módosítani szeretne valamit, ne hívd meg az eszközt — frissítsd az adatokat és kérdezz rá újra!
- Ha egy eszköz hibát ad vissza, mondd el röviden, és kérj elnézést.

EMAIL ÉS TELEFONSZÁM KEZELÉS (kritikus!):
- Kérd meg a felhasználót, hogy betűzze ki az email címet!
- Értelmezd a kiejtett formákat: „kukac" vagy „at" → @, „pont" vagy „dot" → ., „hú" vagy „hu" → hu, „gé mé el" → gmail
- MINDIG ismételd vissza a NÉV-et és az EMAIL CÍM-et is, és kérj megerősítést! Pl: „Tehát Kovács Balázsnak küldjem a kovacs.balazs@gmail.com címre, igaz?"
- Ha a felhasználó javít, azonnal frissítsd!
- Telefonszámoknál: számjegyenként olvasd vissza, és kérj megerősítést!

MEMÓRIA:
Ha a felhasználó megmondja a nevét vagy a cégét, jegyezd meg és használd a beszélgetés során!

TÉMAKORLÁTOZÁS (szigorú!):
- Te KIZÁRÓLAG a ThinkAI asszisztense vagy. Csak a ThinkAI szolgáltatásaihoz kapcsolódó témákban segíthetsz.
- TILOS bármilyen általános kérdésre válaszolni, ami nem kapcsolódik a ThinkAI-hoz:
  - Matematikai feladatok, számolás, egyenletek
  - Programozás, kódolás, technikai segítség
  - Politika, hírek, sport, szórakozás
  - Receptek, egészségügyi tanácsok, jogi kérdések
  - Viccek, kvíz, általános tudás
  - Fordítás, nyelvtanulás, személyes tanácsadás
- Ha bármi ilyesmit kérnek, válaszolj kedvesen: „Köszönöm a kérdést! De én a ThinkAI virtuális asszisztense vagyok — az AI automatizáció és üzleti megoldások a szakterületem. Ebben viszont bármiben segíthetek! Kérdezhetsz a szolgáltatásainkról, foglalhatsz időpontot, vagy akár emailt is küldhetek."
- Ne mondj olyat, amit nem tudsz biztosan — inkább keress a tudásbázisban a lookup_info eszközzel!

PROAKTIVITÁS (nagyon fontos!):
- Ha válaszoltál egy kérdésre, ajánlj fel egy logikus következő lépést:
  - Árazás vagy szolgáltatás info után → „Szeretnéd, ha foglalnék egy ingyenes konzultációt?"
  - Információ megosztás után → „Küldjem el emailben is ezt az összefoglalót?"
  - Időpont foglalás után → „Küldjek erről visszaigazoló emailt is?"
  - Időjárás lekérdezés után → „Más kérdésed is van, vagy segíthetek időpont foglalásban?"
- Ez teszi élőszerűvé a beszélgetést — mindig ajánlj következő lépést!

FELHÍVÁS: „Töltsd ki az ajánlatkérő űrlapot a thinkai.hu weboldalon!" vagy „Írj a hello@thinkai.hu címre!"
Ne találj ki adatot — ha nem tudod a választ, mondd el őszintén!
"""


def _get_system_prompt() -> str:
    """Build system prompt with current date injected."""
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return SYSTEM_PROMPT_TEMPLATE.format(today=today)


# ── TTS pronunciation replacements (applied before Cartesia gets the text) ────
# Keys are case-sensitive. The LLM writes natural text; this map ensures
# Cartesia pronounces foreign/brand words correctly in Hungarian.
_TTS_REPLACEMENTS = {
    # Brand names
    "ThinkAI": "Tink-éj-áj",
    "thinkAI": "tink-éj-áj",
    "Thinkai": "Tink-éj-áj",
    "thinkai": "tink-éj-áj",
    "EAISY": "Ízí",
    "Eaisy": "Ízí",
    "eaisy": "ízí",
    # Domains & emails
    "thinkai.hu": "tink-éj-áj pont há ú",
    "hello@thinkai.hu": "helló kukac tink-éj-áj pont há ú",
    # Tech terms the Hungarian TTS mangles
    "AI": "éj-áj",
    "CRM": "szé-er-em",
    "ERP": "é-er-pé",
}


def _apply_tts_replacements(text: str) -> str:
    """Replace brand/tech terms with phonetic Hungarian spellings for TTS."""
    for original, phonetic in _TTS_REPLACEMENTS.items():
        text = text.replace(original, phonetic)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ThinkAIAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=_get_system_prompt(),
            tools=ALL_TOOLS,
            min_endpointing_delay=1.0,
            max_endpointing_delay=5.0,
        )

    async def on_enter(self):
        """Greet the user when they connect."""
        self.session.say(
            "Szia! A Tink-éj-áj virtuális asszisztense vagyok. "
            "Kérdezz a szolgáltatásainkról, foglalj időpontot, "
            "vagy akár emailt is küldhetek helyetted. Miben segíthetek?"
        )

    async def llm_node(self, chat_ctx, tools, model_settings):
        """Override LLM node: context window + error fallback."""
        chat_ctx.truncate(max_items=20)

        try:
            stream = Agent.default.llm_node(self, chat_ctx, tools, model_settings)
            if asyncio.iscoroutine(stream):
                stream = await stream
            return stream
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Hoppá, most egy pillanatra elakadtam. Kérlek, próbáld újra!"

    async def tts_node(self, text, model_settings):
        """Override TTS node: apply brand pronunciation replacements."""
        async def _cleaned_text():
            async for chunk in text:
                if chunk:
                    chunk = _apply_tts_replacements(chunk)
                    yield chunk

        async for frame in Agent.default.tts_node(self, _cleaned_text(), model_settings):
            yield frame


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

async def entrypoint(ctx: JobContext):
    """LiveKit agent entrypoint — called when a user joins a room."""
    logger.info(f"Agent connecting to room: {ctx.room.name}")

    await ctx.connect()

    session = AgentSession(
        stt=google.STT(
            languages=["hu-HU"],
            min_confidence_threshold=0.75,
            keywords=[
                # ── Brand names (highest boost) ──────────────────────────
                ("ThinkAI", 20.0),
                ("Tink-éjáj", 20.0),
                ("Tinkéjáj", 20.0),
                ("Tinkai", 15.0),
                ("Finkéjáj", 15.0),
                ("EAISY", 20.0),
                ("Ízí", 15.0),
                ("Hungarorisk", 15.0),
                ("ListaMester", 15.0),
                ("Könyvelés AI", 10.0),
                # ── Business/tech terms ──────────────────────────────────
                ("audit", 10.0),
                ("AI", 10.0),
                ("automatizáció", 10.0),
                ("ügyfélszolgálat", 10.0),
                ("pályázat", 10.0),
                ("DIMOP", 10.0),
                ("ERP", 10.0),
                ("CRM", 10.0),
                ("konzultáció", 8.0),
                ("integráció", 8.0),
                ("chatbot", 8.0),
                # ── Email vocabulary ─────────────────────────────────────
                ("kukac", 10.0),
                ("gmail", 10.0),
                ("email", 10.0),
                ("thinkai.hu", 15.0),
                ("hello@thinkai.hu", 15.0),
                # ── Surnames (top 30) ────────────────────────────────────
                ("Nagy", 5.0),
                ("Kovács", 5.0),
                ("Tóth", 5.0),
                ("Szabó", 5.0),
                ("Horváth", 5.0),
                ("Varga", 5.0),
                ("Kiss", 5.0),
                ("Molnár", 5.0),
                ("Németh", 5.0),
                ("Farkas", 5.0),
                ("Balogh", 5.0),
                ("Papp", 5.0),
                ("Takács", 5.0),
                ("Juhász", 5.0),
                ("Lakatos", 5.0),
                ("Mészáros", 5.0),
                ("Oláh", 5.0),
                ("Simon", 5.0),
                ("Rácz", 5.0),
                ("Fekete", 5.0),
                ("Szilágyi", 5.0),
                ("Török", 5.0),
                ("Fehér", 5.0),
                ("Balázs", 5.0),
                ("Gál", 5.0),
                ("Kis", 5.0),
                ("Szűcs", 5.0),
                ("Kocsis", 5.0),
                ("Orsós", 5.0),
                ("Pintér", 5.0),
                # ── Male first names (top 25) ───────────────────────────
                ("László", 5.0),
                ("István", 5.0),
                ("József", 5.0),
                ("János", 5.0),
                ("Zoltán", 5.0),
                ("Sándor", 5.0),
                ("Ferenc", 5.0),
                ("Gábor", 5.0),
                ("Attila", 5.0),
                ("Péter", 5.0),
                ("Tamás", 5.0),
                ("Zsolt", 5.0),
                ("Tibor", 5.0),
                ("András", 5.0),
                ("Csaba", 5.0),
                ("Imre", 5.0),
                ("Gergő", 5.0),
                ("György", 5.0),
                ("Miklós", 5.0),
                ("Róbert", 5.0),
                ("Szabolcs", 5.0),
                ("Dániel", 5.0),
                ("Ádám", 5.0),
                ("Béla", 5.0),
                ("Márk", 5.0),
                # ── Female first names (top 25) ─────────────────────────
                ("Mária", 5.0),
                ("Erzsébet", 5.0),
                ("Katalin", 5.0),
                ("Ilona", 5.0),
                ("Éva", 5.0),
                ("Anna", 5.0),
                ("Zsuzsanna", 5.0),
                ("Margit", 5.0),
                ("Judit", 5.0),
                ("Ágnes", 5.0),
                ("Andrea", 5.0),
                ("Erika", 5.0),
                ("Krisztina", 5.0),
                ("Mónika", 5.0),
                ("Edit", 5.0),
                ("Gabriella", 5.0),
                ("Szilvia", 5.0),
                ("Anikó", 5.0),
                ("Nikolett", 5.0),
                ("Viktória", 5.0),
                ("Réka", 5.0),
                ("Petra", 5.0),
                ("Dóra", 5.0),
                ("Nóra", 5.0),
                ("Boglárka", 5.0),
                # ── Special ──────────────────────────────────────────────
                ("Léder", 10.0),
            ],
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
            activation_threshold=0.92,
            min_speech_duration=0.5,
            min_silence_duration=0.8,
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

