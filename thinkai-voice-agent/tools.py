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
CALENDAR_FILE = THIS_DIR / "calendar.json"
EMAILS_FILE = THIS_DIR / "emails.json"


# ── JSON helpers ─────────────────────────────────────────────────────────────
def _read_json(path: Path) -> list:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _write_json(path: Path, data: list):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SEND FOLLOW-UP EMAIL (Brevo Transactional API) — also logs to emails.json
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
    raw_key = os.getenv("BREVO_API_KEY", "")
    # Try raw key first. If it looks base64-encoded (no hyphens, starts with 'ey'), try decoding.
    api_key = raw_key
    if raw_key and not raw_key.startswith("xkeysib-"):
        try:
            import base64 as b64module
            decoded = b64module.b64decode(raw_key).decode()
            parsed = json.loads(decoded)
            api_key = parsed.get("api_key", raw_key)
            logger.info("Brevo key: decoded from base64/JSON")
        except Exception:
            api_key = raw_key
    logger.info(f"Brevo key starts with: {api_key[:12]}...")
    logger.info(f"Sending follow-up email to {recipient_name} <{recipient_email}>")

    sent_ok = False
    error_msg = ""

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
            sent_ok = True
    except Exception as e:
        logger.error(f"Email error: {e}")
        error_msg = str(e)

    # Log the email regardless of send success
    emails = _read_json(EMAILS_FILE)
    emails.append({
        "id": len(emails) + 1,
        "to_name": recipient_name,
        "to_email": recipient_email,
        "subject": subject,
        "message": message,
        "sent_at": datetime.utcnow().isoformat(),
        "status": "sent" if sent_ok else "failed",
        "error": error_msg if not sent_ok else None,
    })
    _write_json(EMAILS_FILE, emails)

    if sent_ok:
        return f"Email sikeresen elküldve {recipient_name} ({recipient_email}) részére."
    else:
        return f"Hiba az email küldésekor: {error_msg}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHECK CALENDAR (local JSON store)
# ═══════════════════════════════════════════════════════════════════════════════

@function_tool(description="Naptár ellenőrzése: megnézi, milyen események vannak a következő napokban. Használd, ha a felhasználó időpontot keres vagy tudni akarja, mikor szabad a naptár.")
async def check_calendar(
    ctx: RunContext,
    days_ahead: Annotated[int, "Hány napra előre nézze a naptárat (alapértelmezett: 7)"] = 7,
) -> str:
    """Naptár ellenőrzése a következő napokra."""
    logger.info(f"Checking calendar for next {days_ahead} days")

    events = _read_json(CALENDAR_FILE)
    if not events:
        return f"A következő {days_ahead} napban nincsenek rögzített események — teljesen szabad a naptár!"

    now = datetime.utcnow()
    cutoff = now + timedelta(days=days_ahead)

    upcoming = []
    for ev in events:
        try:
            ev_dt = datetime.fromisoformat(ev["start"])
            if now <= ev_dt <= cutoff:
                upcoming.append(ev)
        except Exception:
            continue

    upcoming.sort(key=lambda e: e["start"])

    if not upcoming:
        return f"A következő {days_ahead} napban nincsenek rögzített események — teljesen szabad a naptár!"

    event_list = []
    for ev in upcoming[:10]:
        try:
            dt = datetime.fromisoformat(ev["start"])
            formatted = dt.strftime("%m/%d %H:%M")
        except Exception:
            formatted = ev["start"]
        title = ev.get("title", "Névtelen esemény")
        duration = ev.get("duration_minutes", 30)
        event_list.append(f"- {formatted}: {title} ({duration} perc)")

    return f"A következő {days_ahead} napban {len(upcoming)} esemény van:\n" + "\n".join(event_list)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BOOK A MEETING (local JSON store)
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
    logger.info(f"Booking meeting: {title} on {date} at {time}")

    try:
        start_dt = datetime.fromisoformat(f"{date}T{time}:00")
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        events = _read_json(CALENDAR_FILE)
        new_id = max((e.get("id", 0) for e in events), default=0) + 1
        events.append({
            "id": new_id,
            "title": title,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "duration_minutes": duration_minutes,
            "attendee": attendee_email or None,
            "created_at": datetime.utcnow().isoformat(),
        })
        _write_json(CALENDAR_FILE, events)

        result = f"Találkozó sikeresen lefoglalva: {title}, {date} {time}-kor, {duration_minutes} perces."
        if attendee_email:
            result += f" Meghívott: {attendee_email}."
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
        tasks = _read_json(TASKS_FILE)
        new_task = {
            "id": len(tasks) + 1,
            "text": task,
            "priority": priority,
            "due_date": due_date,
            "created_at": datetime.utcnow().isoformat(),
            "completed": False,
        }
        tasks.append(new_task)
        _write_json(TASKS_FILE, tasks)

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


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MODIFY CALENDAR EVENT (voice command)
# ═══════════════════════════════════════════════════════════════════════════════

@function_tool(description="Naptári esemény módosítása. Használd, ha a felhasználó meg akarja változtatni egy meglévő találkozó időpontját, címét vagy időtartamát.")
async def modify_meeting(
    ctx: RunContext,
    event_title: Annotated[str, "A módosítandó esemény címe (vagy egy része, ami azonosítja)"],
    new_title: Annotated[str, "Az új cím (ha változik, különben hagyd üresen)"] = "",
    new_date: Annotated[str, "Az új dátum YYYY-MM-DD formátumban (ha változik)"] = "",
    new_time: Annotated[str, "Az új időpont HH:MM formátumban (ha változik)"] = "",
    new_duration_minutes: Annotated[int, "Az új időtartam percben (ha változik)"] = 0,
) -> str:
    """Naptári esemény módosítása."""
    logger.info(f"Modifying meeting: {event_title}")

    events = _read_json(CALENDAR_FILE)
    if not events:
        return "Nincs egyetlen esemény sem a naptárban."

    # Find the event by title (fuzzy match)
    found = None
    for ev in events:
        if event_title.lower() in ev.get("title", "").lower():
            found = ev
            break

    if not found:
        titles = ", ".join(e.get("title", "?") for e in events)
        return f"Nem találtam ilyen eseményt. A naptárban ezek vannak: {titles}"

    try:
        if new_title:
            found["title"] = new_title
        if new_date or new_time:
            old_dt = datetime.fromisoformat(found["start"])
            d = new_date or old_dt.strftime("%Y-%m-%d")
            t = new_time or old_dt.strftime("%H:%M")
            new_start = datetime.fromisoformat(f"{d}T{t}:00")
            dur = new_duration_minutes or found.get("duration_minutes", 30)
            found["start"] = new_start.isoformat()
            found["end"] = (new_start + timedelta(minutes=dur)).isoformat()
            found["duration_minutes"] = dur
        elif new_duration_minutes:
            start = datetime.fromisoformat(found["start"])
            found["duration_minutes"] = new_duration_minutes
            found["end"] = (start + timedelta(minutes=new_duration_minutes)).isoformat()

        _write_json(CALENDAR_FILE, events)

        changes = []
        if new_title: changes.append(f"cím: {new_title}")
        if new_date: changes.append(f"dátum: {new_date}")
        if new_time: changes.append(f"idő: {new_time}")
        if new_duration_minutes: changes.append(f"időtartam: {new_duration_minutes} perc")
        return f"Esemény módosítva ({found['title']}): {', '.join(changes)}."
    except Exception as e:
        logger.error(f"Modify error: {e}")
        return f"Hiba a módosításkor: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DELETE CALENDAR EVENT (voice command)
# ═══════════════════════════════════════════════════════════════════════════════

@function_tool(description="Naptári esemény törlése. Használd, ha a felhasználó le akarja mondani vagy törölni akar egy találkozót.")
async def delete_meeting(
    ctx: RunContext,
    event_title: Annotated[str, "A törlendő esemény címe (vagy egy része, ami azonosítja)"],
) -> str:
    """Naptári esemény törlése."""
    logger.info(f"Deleting meeting: {event_title}")

    events = _read_json(CALENDAR_FILE)
    if not events:
        return "Nincs egyetlen esemény sem a naptárban."

    # Find and remove
    original_count = len(events)
    events = [e for e in events if event_title.lower() not in e.get("title", "").lower()]

    if len(events) == original_count:
        titles = ", ".join(e.get("title", "?") for e in events)
        return f"Nem találtam ilyen eseményt. A naptárban ezek vannak: {titles}"

    _write_json(CALENDAR_FILE, events)
    return f"Esemény törölve: {event_title}."


# All tools for easy import
ALL_TOOLS = [
    send_followup_email,
    check_calendar,
    book_meeting,
    modify_meeting,
    delete_meeting,
    get_weather,
    create_task,
    lookup_info,
]
