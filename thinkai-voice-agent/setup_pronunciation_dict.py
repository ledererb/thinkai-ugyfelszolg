#!/usr/bin/env python3
"""
One-time setup script: Create a Cartesia pronunciation dictionary for ThinkAI.
Run this once, then add the CARTESIA_PRONUNCIATION_DICT_ID to your .env file.

Usage:
    python setup_pronunciation_dict.py
"""

import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

API_KEY = os.getenv("CARTESIA_API_KEY")
if not API_KEY:
    print("ERROR: Set CARTESIA_API_KEY in .env")
    exit(1)

BASE_URL = "https://api.cartesia.ai"
HEADERS = {
    "X-API-Key": API_KEY,
    "Cartesia-Version": "2024-06-10",
    "Content-Type": "application/json",
}

# Pronunciation rules: word → how TTS should say it
RULES = [
    {"alias": "ThinkAI", "phoneme": "Tink-éj-áj"},
    {"alias": "thinkai", "phoneme": "tink-éj-áj"},
    {"alias": "EAISY", "phoneme": "Ízí"},
    {"alias": "thinkai.hu", "phoneme": "tink-éj-áj pont há ú"},
    {"alias": "hello@thinkai.hu", "phoneme": "helló kukac tink-éj-áj pont há ú"},
]


def main():
    # Check existing dictionaries
    print("Checking existing dictionaries...")
    resp = requests.get(f"{BASE_URL}/pronunciation-dicts/", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    existing = resp.json()

    # Check if "thinkai-pronunciations" already exists
    for d in existing:
        if d.get("name") == "thinkai-pronunciations":
            print(f"Dictionary already exists: {d['id']}")
            print(f"Add to .env: CARTESIA_PRONUNCIATION_DICT_ID={d['id']}")
            return

    # Create new dictionary
    print("Creating pronunciation dictionary...")
    payload = {
        "name": "thinkai-pronunciations",
        "rules": RULES,
    }
    resp = requests.post(
        f"{BASE_URL}/pronunciation-dicts/",
        headers=HEADERS,
        json=payload,
        timeout=10,
    )
    resp.raise_for_status()
    result = resp.json()

    dict_id = result["id"]
    print(f"Created dictionary: {dict_id}")
    print(f"\nAdd this to your .env and Railway env vars:")
    print(f"CARTESIA_PRONUNCIATION_DICT_ID={dict_id}")


if __name__ == "__main__":
    main()
