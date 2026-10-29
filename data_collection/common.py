import time
import requests

BASE = "https://cafef.vn/du-lieu"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (research; thesis project)"}
DELAY = 1.0  # seconds between requests


def get(url, params=None, retries=8):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            time.sleep(DELAY)
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(min(30 * (attempt + 1), 180))
