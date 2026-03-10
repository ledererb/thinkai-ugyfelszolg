"""
ThinkAI Voice Agent — LiveKit Agents Server
Real-time voice assistant powered by LiveKit + Google STT + Gemini 2.5 Flash + Cartesia TTS
Dynamic HU/EN language switching via LLM detection
"""

import asyncio
import os
import re
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
- SOHA ne kezdj két egymást követő választ ugyanazzal a szóval
- Ha problémát hallasz, először mutass empátiát, aztán oldd meg

NYELV:
- Alapértelmezett: magyar
- Ha angolul szólalnak meg, válaszolj angolul
- Kövesd a felhasználó nyelvét: ha váltanak, te is válts
- NE keverd a nyelveket egy válaszon belül

NYELVVÁLTÁS JELZÉS (KRITIKUS!):
- Ha a felhasználó ANGOLRA vált, kezdd a válaszod ezzel: [LANG:EN]
- Ha a felhasználó VISSZAVÁLT magyarra, kezdd a válaszod ezzel: [LANG:HU]
- Ha nincs nyelvváltás, NE használd a jelzést — csak változáskor!
- A [LANG:XX] jelzést a felhasználó NEM fogja hallani, csak belső használatra van

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
- MIELŐTT bármilyen eszközt meghívnál, FOGLALD ÖSSZE amit tenni fogsz és kérdezd meg: "Jól értettem? Mehet?"
- CSAK "igen", "ja", "mehet", "OK", "yes" válasz után hívd meg az eszközt!
- Ha a felhasználó bármit módosít, NE hívd meg az eszközt — frissítsd az adatokat és kérdezz újra!
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

GUARDRAILS:
- Ha a felhasználó olyat kérdez, amihez nincs közöd (politika, sport, személyes tanácsok, viccek stb.), udvariasan tereld vissza: "Érdekes kérdés! De nekem a szakterületem az AI automatizáció és üzleti megoldások. Ebben tudok igazán segíteni!"
- NE mondj olyat, amit nem tudsz biztosan — inkább ajánld a lookup_info eszközt

PROAKTIVITÁS (nagyon fontos a demo-hoz!):
- Ha válaszoltál egy kérdésre, ajánlj fel egy logikus következő lépést:
  - Árazás vagy szolgáltatás info után → "Szeretnéd, ha foglalnék egy ingyenes konzultációt?"
  - Információ megosztás után → "Küldjem el emailben is ezt az összefoglalót?"
  - Időpont foglalás után → "Küldjek erről visszaigazoló emailt is?"
  - Időjárás után → "Más kérdésed is van, vagy segíthetek időpont foglalásban?"
- EZ TESZI ÉLŐSZERŰVÉ a beszélgetést — mindig ajánlj következő lépést!

CTA: "Töltsd ki az ajánlatkérő űrlapot a tink-éj-áj pont há ú weboldalon!" vagy "Írj a helló kukac tink-éj-áj pont há ú címre!"
Ne találj ki adatot — ha nem tudod, mondd el őszintén!
"""


def _get_system_prompt() -> str:
    """Build system prompt with current date injected."""
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return SYSTEM_PROMPT_TEMPLATE.format(today=today)


# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION — switch STT/TTS dynamically
# ═══════════════════════════════════════════════════════════════════════════════

_LANG_TAG_RE = re.compile(r"\[LANG:(HU|EN)\]")


def _switch_language(new_lang: str, session: AgentSession, current_lang: list):
    """Switch STT and TTS language if changed."""
    if new_lang == current_lang[0]:
        return

    current_lang[0] = new_lang
    logger.info(f"🌐 Language switch → {new_lang.upper()}")

    try:
        if new_lang == "en":
            session.stt.update_options(languages=["en-US", "hu-HU"])
        else:
            session.stt.update_options(languages=["hu-HU", "en-US"])
    except Exception as e:
        logger.warning(f"STT language switch failed: {e}")

    try:
        session.tts.update_options(language=new_lang)
    except Exception as e:
        logger.warning(f"TTS language switch failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ThinkAIAgent(Agent):
    def __init__(self):
        self._current_lang = ["hu"]  # mutable for closure
        super().__init__(
            instructions=_get_system_prompt(),
            tools=ALL_TOOLS,
            min_endpointing_delay=0.6,
            max_endpointing_delay=3.0,
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
        """Override TTS node: detect [LANG:XX] tags, switch BEFORE creating stream.

        Critical: Cartesia copies tts._opts at stream creation (`replace(tts._opts)`).
        Agent.default.tts_node creates the stream immediately, so update_options()
        during text streaming has no effect. We must:
        1. Buffer initial text chunks
        2. Detect and strip [LANG:XX] tag
        3. Call update_options() to switch language
        4. THEN create the TTS stream with the correct language
        """
        activity = self._get_activity_or_raise()
        assert activity.tts is not None

        # Step 1: Buffer first chunks to detect language tag
        first_chunks = []
        remaining_iter = text.__aiter__()
        detected = False

        # Collect chunks until we find a tag or enough text
        async for chunk in remaining_iter:
            first_chunks.append(chunk)
            combined = "".join(first_chunks)

            match = _LANG_TAG_RE.search(combined)
            if match:
                # Found tag — switch language
                _switch_language(
                    match.group(1).lower(),
                    self.session,
                    self._current_lang,
                )
                # Strip tag from combined text
                combined = _LANG_TAG_RE.sub("", combined).lstrip()
                first_chunks = [combined] if combined else []
                detected = True
                break
            elif len(combined) >= 12 or (combined and not combined.lstrip().startswith("[")):
                # No tag — stop buffering
                break

        # Step 2: Create async generator with buffered + remaining text
        async def _cleaned_text():
            # Yield buffered chunks first
            for chunk in first_chunks:
                if chunk:
                    yield chunk

            # Then yield remaining chunks, stripping any late tags
            async for chunk in remaining_iter:
                match = _LANG_TAG_RE.search(chunk)
                if match:
                    _switch_language(
                        match.group(1).lower(),
                        self.session,
                        self._current_lang,
                    )
                    chunk = _LANG_TAG_RE.sub("", chunk).lstrip()
                if chunk:
                    yield chunk

        # Step 3: NOW create TTS stream (language is already set correctly)
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
            languages=["hu-HU", "en-US"],
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
                # ── Email vocabulary ─────────────────────────────────────
                ("kukac", 10.0),
                ("gmail", 10.0),
                ("email", 10.0),
                ("thinkai.hu", 15.0),
                ("hello@thinkai.hu", 15.0),
                # ── Common Hungarian names ───────────────────────────────
                ("Balázs", 5.0),
                ("Gergő", 5.0),
                ("László", 5.0),
                ("Szabolcs", 5.0),
                ("Tamás", 5.0),
                ("Péter", 5.0),
                ("János", 5.0),
                ("András", 5.0),
                ("Zoltán", 5.0),
                ("István", 5.0),
                ("Attila", 5.0),
                ("Nagy", 5.0),
                ("Kovács", 5.0),
                ("Tóth", 5.0),
                ("Szabó", 5.0),
                ("Horváth", 5.0),
                ("Varga", 5.0),
                ("Kiss", 5.0),
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
            activation_threshold=0.85,
            min_speech_duration=0.4,
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

