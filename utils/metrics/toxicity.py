import os
import threading
import time
import requests
from dotenv import load_dotenv


load_dotenv()

PERSPECTIVE_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
PERSPECTIVE_ATTRIBUTES = [
    "TOXICITY",
    "SEVERE_TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "PROFANITY",
    "THREAT",
]
_PERSPECTIVE_SEMAPHORE = threading.BoundedSemaphore(int(os.getenv("PERSPECTIVE_MAX_CONCURRENCY", "1")))


def get_perspective_api_key():
    api_key = os.getenv("PERSPECTIVE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("Valid Perspective API key not found. Check PERSPECTIVE_API_KEY in .env.")
    return api_key


def analyze_text_toxicity(text):
    """
    Calls Perspective API and returns toxicity scores keyed by attribute name.
    """
    timeout = float(os.getenv("PERSPECTIVE_TIMEOUT", "60"))
    max_attempts = int(os.getenv("PERSPECTIVE_MAX_ATTEMPTS", "5"))
    payload = {
        "comment": {"text": text or ""},
        "requestedAttributes": {attr: {} for attr in PERSPECTIVE_ATTRIBUTES},
        "languages": ["en"],
        "doNotStore": True,
    }

    for attempt in range(1, max_attempts + 1):
        try:
            with _PERSPECTIVE_SEMAPHORE:
                response = requests.post(
                    PERSPECTIVE_URL,
                    params={"key": get_perspective_api_key()},
                    json=payload,
                    timeout=timeout,
                )
            response.raise_for_status()
            result = response.json()
            return {
                attr: data["summaryScore"]["value"]
                for attr, data in result.get("attributeScores", {}).items()
            }
        except Exception as e:
            print(f"Perspective API request failed ({attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
    return None
