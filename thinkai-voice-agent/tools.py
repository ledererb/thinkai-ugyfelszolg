"""
ThinkAI Voice Agent — Tool Implementations (LiveKit Agents v1.4)
Function tools using @function_tool decorator for the voice assistant.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import httpx
from livekit.agents import function_tool, RunContext
from loguru import logger

# ── Paths ────────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
TASKS_FILE = THIS_DIR / "tasks.json"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SEND FOLLOW-UP EMAIL (Brevo Transactional API)
# ═══════════════════════════════════════════════════════════════════════════════

@function_tool(description="Follow-up email küldése egy érdeklődőnek vagy ügyfélnek. Használd, ha a felhasználó emailt szeretne küldeni valakinek.")
async def send_followup_email(
    ctx: RunContext,
    recipient_name: Annotated[str, "A címzett neve"],
    recipient_email: Annotated[str, "A címzett email címe"],
    message: Annotated[str, "Az email szövegtörzse (rövid, barátságos, szakmai)"],
    subject: Annotated[str, "Az email tárgya"] = "ThinkAI — Köszönjük érdeklődését!",
) -> str:
    """Follow-up email küldése egy érdeklődőnek."""
    import base64 as b64module
    raw_key = os.getenv("BREVO_API_KEY", "")
    # Decode base64-wrapped MCP key if needed
    try:
        decoded = b64module.b64decode(raw_key).decode()
        import json as _json
        api_key = _json.loads(decoded).get("api_key", raw_key)
    except Exception:
        api_key = raw_key
    logger.info(f"Sending follow-up email to {recipient_name} <{recipient_email}>")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.brevo.com/v3/smtp/email",
                headers={"api-key": api_key, "Content-Type": "application/json"},
                json={
                    "sender": {"name": "ThinkAI", "email": "hello@thinkai.hu"},
                    "to": [{"email": recipient_email, "name": recipient_name}],
                    "subject": subject,
                    "htmlContent": f"""
                    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                        <h2 style="color: #1a1a2e;">Kedves {recipient_name}!</h2>
                        <p>{message}</p>
                        <hr style="border: 1px solid #eee; margin: 20px 0;">
                        <p style="color: #666; font-size: 14px;">
                            Üdvözlettel,<br>
                            <strong>ThinkAI csapat</strong><br>
                            <a href="https://thinkai.hu">thinkai.hu</a> | hello@thinkai.hu
                        </p>
                    </div>
                    """,
                },
                timeout=10,
            )
            resp.raise_for_status()
            return f"Email sikeresen elküldve {recipient_name} ({recipient_email}) részére."
    except Exception as e:
        logger.error(f"Email error: {e}")
        return f"Hiba az email küldésekor: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHECK CALENDAR AVAILABILITY (Google Calendar API)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_google_calendar_service():
    """Create a Google Calendar service client."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    creds_path = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(THIS_DIR / "google-credentials.json"),
    )

    if creds_json:
        info = json.loads(creds_json)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/calendar"]
        )
    else:
        creds = service_account.Credentials.from_service_account_file(
            creds_path, scopes=["https://www.googleapis.com/auth/calendar"]
        )

    return build("calendar", "v3", credentials=creds)


@function_tool(description="Naptár ellenőrzése: megnézi, milyen események vannak a következő napokban. Használd, ha a felhasználó időpontot keres vagy tudni akarja, mikor szabad a naptár.")
async def check_calendar(
    ctx: RunContext,
    days_ahead: Annotated[int, "Hány napra előre nézze a naptárat (alapértelmezett: 5)"] = 5,
) -> str:
    """Naptár ellenőrzése a következő napokra."""
    import asyncio

    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    logger.info(f"Checking calendar for next {days_ahead} days")

    try:
        service = _get_google_calendar_service()
        now = datetime.utcnow()
        time_min = now.isoformat() + "Z"
        time_max = (now + timedelta(days=days_ahead)).isoformat() + "Z"

        loop = asyncio.get_event_loop()
        events_result = await loop.run_in_executor(
            None,
            lambda: service.events()
            .list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute(),
        )

        events = events_result.get("items", [])
        if not events:
            return f"A következő {days_ahead} napban nincsenek rögzített események — teljesen szabad a naptár!"

        event_list = []
        for event in events[:10]:
            start = event["start"].get("dateTime", event["start"].get("date"))
            summary = event.get("summary", "Névtelen esemény")
            try:
                dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                formatted = dt.strftime("%m/%d %H:%M")
            except Exception:
                formatted = start
            event_list.append(f"- {formatted}: {summary}")

        return f"A következő {days_ahead} napban {len(events)} esemény van:\n" + "\n".join(event_list)
    except Exception as e:
        logger.error(f"Calendar error: {e}")
        return f"Hiba a naptár lekérdezésekor: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BOOK A MEETING (Google Calendar API)
# ═══════════════════════════════════════════════════════════════════════════════

@function_tool(description="Találkozó/meeting foglalása a naptárba. Használd, ha a felhasználó időpontot szeretne foglalni.")
async def book_meeting(
    ctx: RunContext,
    title: Annotated[str, "A meeting címe/témája"],
    date: Annotated[str, "A meeting dátuma YYYY-MM-DD formátumban"],
    time: Annotated[str, "A meeting kezdési időpontja HH:MM formátumban (pl. 10:00)"],
    duration_minutes: Annotated[int, "A meeting hossza percben"] = 30,
    attendee_email: Annotated[str, "A meghívott email címe (opcionális)"] = "",
) -> str:
    """Találkozó foglalása a naptárba."""
    import asyncio

    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    logger.info(f"Booking meeting: {title} on {date} at {time}")

    try:
        service = _get_google_calendar_service()
        start_dt = datetime.fromisoformat(f"{date}T{time}:00")
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        event = {
            "summary": title,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": "Europe/Budapest"},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": "Europe/Budapest"},
        }
        if attendee_email:
            event["attendees"] = [{"email": attendee_email}]

        loop = asyncio.get_event_loop()
        created = await loop.run_in_executor(
            None,
            lambda: service.events()
            .insert(calendarId=calendar_id, body=event, sendUpdates="all")
            .execute(),
        )

        result = f"Találkozó sikeresen lefoglalva: {title}, {date} {time}-kor, {duration_minutes} perces."
        if attendee_email:
            result += f" Meghívó elküldve: {attendee_email}."
        return result
    except Exception as e:
        logger.error(f"Booking error: {e}")
        return f"Hiba a találkozó foglalásakor: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WEATHER CHECK (Open-Meteo API — no API key needed!)
# ═══════════════════════════════════════════════════════════════════════════════

CITY_COORDS = {
    "budapest": (47.4979, 19.0402),
    "debrecen": (47.5316, 21.6273),
    "szeged": (46.253, 20.1414),
    "miskolc": (48.1035, 20.7784),
    "pécs": (46.0727, 18.2323),
    "győr": (47.6875, 17.6504),
    "nyíregyháza": (47.9553, 21.7174),
    "kecskemét": (46.8964, 19.6897),
    "székesfehérvár": (47.1860, 18.4221),
    "vienna": (48.2082, 16.3738),
    "bécs": (48.2082, 16.3738),
    "london": (51.5074, -0.1278),
    "new york": (40.7128, -74.0060),
    "paris": (48.8566, 2.3522),
    "párizs": (48.8566, 2.3522),
    "berlin": (52.5200, 13.4050),
}


@function_tool(description="Aktuális időjárás lekérdezése egy városban. Használd, ha a felhasználó az időjárásról kérdez.")
async def get_weather(
    ctx: RunContext,
    city: Annotated[str, "A város neve (pl. Budapest, Debrecen, Bécs)"],
) -> str:
    """Időjárás lekérdezése."""
    city_lower = city.lower().strip()
    coords = CITY_COORDS.get(city_lower, CITY_COORDS["budapest"])
    if city_lower not in CITY_COORDS:
        city = "Budapest"
    lat, lon = coords

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon, "current_weather": "true", "timezone": "Europe/Budapest"},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()

        weather = data.get("current_weather", {})
        temp = weather.get("temperature", "?")
        wind = weather.get("windspeed", "?")
        code = weather.get("weathercode", 0)

        weather_desc = {
            0: "tiszta égbolt", 1: "enyhén felhős", 2: "részben felhős",
            3: "borult", 45: "ködös", 48: "zúzmarás köd",
            51: "enyhe szitálás", 53: "mérsékelt szitálás", 55: "sűrű szitálás",
            61: "enyhe eső", 63: "mérsékelt eső", 65: "erős eső",
            71: "enyhe havazás", 73: "mérsékelt havazás", 75: "erős havazás",
            80: "enyhe zápor", 81: "mérsékelt zápor", 82: "erős zápor",
            95: "zivatar", 96: "jégesős zivatar", 99: "erős jégesős zivatar",
        }.get(code, "ismeretlen")

        return f"{city.title()}: {temp}°C, {weather_desc}, szél {wind} km/h."
    except Exception as e:
        logger.error(f"Weather error: {e}")
        return f"Hiba az időjárás lekérdezésekor: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CREATE TASK/NOTE (local JSON store)
# ═══════════════════════════════════════════════════════════════════════════════

@function_tool(description="Feladat/teendő/jegyzet rögzítése. Használd, ha a felhasználó jegyezni akar valamit, vagy feladatot szeretne rögzíteni.")
async def create_task(
    ctx: RunContext,
    task: Annotated[str, "A feladat szövege"],
    priority: Annotated[str, "Prioritás: low/normal/high"] = "normal",
    due_date: Annotated[str, "Határidő YYYY-MM-DD formátumban (opcionális)"] = "",
) -> str:
    """Feladat rögzítése."""
    logger.info(f"Creating task: {task}")

    try:
        tasks = []
        if TASKS_FILE.exists():
            tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))

        new_task = {
            "id": len(tasks) + 1,
            "text": task,
            "priority": priority,
            "due_date": due_date,
            "created_at": datetime.utcnow().isoformat(),
            "completed": False,
        }
        tasks.append(new_task)
        TASKS_FILE.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

        result = f'Feladat rögzítve: "{task}"'
        if due_date:
            result += f" — határidő: {due_date}"
        return result + "."
    except Exception as e:
        logger.error(f"Task error: {e}")
        return f"Hiba a feladat mentésekor: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. KNOWLEDGE LOOKUP (structured ThinkAI info)
# ═══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_BASE = {
    "pricing": (
        "Az árazás projektfüggő. Általánosságban: "
        "Audit díja: 150.000 – 300.000 Ft (100% pénzvisszafizetési garanciával). "
        "Egyedi fejlesztés: projekttől függően 500.000 Ft-tól. "
        "AI-ügyfélszolgálat: havi előfizetéses modell, a hívásszámtól függően. "
        "Pontos árajánlatért töltse ki az ajánlatkérő űrlapot a thinkai.hu weboldalon."
    ),
    "audit": (
        "Az audit egy teljeskörű szervezeti átvilágítás, ahol feltérképezzük a vállalkozás "
        "folyamatait és megtaláljuk azokat a pontokat, ahol az AI azonnal értéket teremthet. "
        "100% pénzvisszafizetési garancia: ha nem tetszik az audit eredménye, kérdés nélkül "
        "visszafizetjük az árát."
    ),
    "tech_stack": (
        "A ThinkAI csapat a következő technológiákat használja: "
        "Make.com és n8n workflow automatizáció, Python/Node.js backend fejlesztés, "
        "OpenAI, Anthropic Claude, Google AI modellek, "
        "egyedi AI-agentek fejlesztése, ERP integráció, CRM automatizáció."
    ),
    "team": (
        "A ThinkAI egy magyar startup, amely tapasztalt AI és szoftverfejlesztő "
        "szakemberekből áll. A csapat célja, hogy a legmodernebb AI megoldásokat "
        "tegye elérhetővé a magyar kis- és középvállalkozások számára."
    ),
    "eaisy": (
        "Az EAISY a ThinkAI saját fejlesztésű termékcsaládja: moduláris ERP és AI "
        "eszközök, amelyek azonnal integrálhatók a mindennapi működésbe."
    ),
    "guarantee": (
        "A ThinkAI 100% pénzvisszafizetési garanciát ad az audit szolgáltatásra. "
        "Ha nem tetszik az audit eredménye, kérdés nélkül visszafizetjük az árát."
    ),
}


@function_tool(description="ThinkAI belső tudásbázis lekérdezése. Használd, ha a felhasználó részletes információt kér az árazásról, auditról, technológiáról, csapatról, EAISY-ról vagy a garanciáról.")
async def lookup_info(
    ctx: RunContext,
    topic: Annotated[str, "A keresett téma: pricing, audit, tech_stack, team, eaisy, guarantee"],
) -> str:
    """ThinkAI tudásbázis lekérdezése."""
    topic_lower = topic.lower().strip()
    logger.info(f"Knowledge lookup: {topic_lower}")

    if topic_lower in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[topic_lower]

    # Fuzzy match
    for key, value in KNOWLEDGE_BASE.items():
        if key in topic_lower or topic_lower in key:
            return value

    topics = ", ".join(KNOWLEDGE_BASE.keys())
    return (
        f"Erről a témáról nincs részletes információm. "
        f"A következő témákban tudok segíteni: {topics}. "
        f"Részletesebb információért keresd a csapatot a hello@thinkai.hu címen!"
    )


# All tools for easy import
ALL_TOOLS = [
    send_followup_email,
    check_calendar,
    book_meeting,
    get_weather,
    create_task,
    lookup_info,
]
