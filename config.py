"""Central configuration for MIDAS. Every tunable lives here so behavior
can be adjusted without touching orchestration, tool, or UI code."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Ollama models -----------------------------------------------------
# Small, single-purpose models rather than one large model, so the 6GB
# VRAM budget only ever has to hold one of them at a time.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

ORCHESTRATOR_MODEL = "qwen2.5:7b"    # general chat + tool calling
ROUTER_MODEL = "arch-router:1.5b"    # single-word traffic classifier
VISION_MODEL = "qwen2.5vl:3b"        # OCR/notification-screenshot extraction
EMBED_MODEL = "nomic-embed-text"     # 768-dim text embeddings for retrieval
# Not used anywhere until core/memory.py — must be pulled once separately:
#   ollama pull nomic-embed-text

# Short keep-alives so idle models evict from VRAM quickly instead of
# lingering and starving whichever model runs next.
ROUTER_KEEP_ALIVE = "30s"
CHAT_KEEP_ALIVE = "2m"
VISION_KEEP_ALIVE = "30s"  # used a handful of times overnight, evict fast
EMBED_KEEP_ALIVE = "30s"   # tiny model, called in short bursts — evict fast

# Aggressive context budgets appropriate for a 6GB card. ROUTER_NUM_CTX
# needs headroom for the full route list (every registered tool's
# description gets concatenated into one classification prompt) — with
# 10 routes this is already close to 1024 tokens, and it only grows as
# more tools are added, so this is sized with real margin rather than
# tightly to today's route count.
ROUTER_NUM_CTX = 2048
CHAT_NUM_CTX = 2048
# A real iPhone screenshot needs far more image tokens than a tiny test
# image — observed 3882 tokens against a real lock-screen capture, so
# budget well above that for headroom across different screen sizes.
VISION_NUM_CTX = 8192
EMBED_NUM_CTX = 2048      # nomic-embed-text's native context window

# Lower than Ollama's default (0.8) — small models stay on-instruction,
# classify routes more consistently, and extract structured arguments
# more reliably at low temperature.
ROUTER_TEMPERATURE = 0.1
CHAT_TEMPERATURE = 0.3

# --- Storage -------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "midas.db"

# --- Embeddings / retrieval ------------------------------------------------
# Vectors are stored as raw little-endian float32 in a SQLite BLOB. This
# dtype is part of the on-disk format: changing it, or changing EMBED_MODEL
# to one with a different dimensionality, invalidates every embedding
# already stored (see core/memory.py).
EMBED_DTYPE = "float32"

# --- Morning briefing ----------------------------------------------------
USER_NAME = "Jonathan"
BRIEFING_TIME = "07:30"  # 24h HH:MM, local time
WATCHLIST_TICKERS = ["SPY", "QQQ", "NVDA"]
NEWS_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.arstechnica.com/arstechnica/index",
]
HEADLINE_LIMIT = 5
REMINDER_POLL_INTERVAL_MINUTES = 1

# --- Weather (Open-Meteo, no API key) --------------------------------------
# PLACEHOLDER COORDINATES — replace with your own before first use.
WEATHER_LAT = 40.7128
WEATHER_LON = -74.0060
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TEMPERATURE_UNIT = "fahrenheit"   # or "celsius"
WEATHER_WIND_SPEED_UNIT = "mph"
WEATHER_TIMEOUT_SECONDS = 8

# --- Wardrobe ---------------------------------------------------------------
WARDROBE_CATEGORIES = ["top", "bottom", "outerwear", "shoes", "accessory"]
# Categories that always appear in a suggested outfit. Outerwear is added
# conditionally (see WARDROBE_OUTERWEAR_MAX_TEMP) rather than listed here.
WARDROBE_OUTFIT_CATEGORIES = ["top", "bottom", "shoes"]

# warmth_level and formality are both 1-5 integer scales:
#   warmth    1 = lightest (tank/shorts)  ... 5 = heaviest (parka)
#   formality 1 = loungewear/gym          ... 5 = suit/formal
# Deterministic temperature -> target warmth: first band whose upper bound
# the current temperature falls at or below wins.
WARDROBE_WARMTH_BANDS = [(32, 5), (45, 4), (60, 3), (75, 2), (999, 1)]
WARDROBE_WARMTH_TOLERANCE = 1      # +/- around the target band
WARDROBE_OUTERWEAR_MAX_TEMP = 60   # suggest outerwear at or below this
WARDROBE_RAIN_PROBABILITY_THRESHOLD = 40  # percent; forces outerwear

# Layering: "top" category items can optionally be tagged with a layer
# role. At or below this temperature, suggest_fit tries to pick one base
# + one mid item instead of a single top — but only if both roles are
# actually represented in the wardrobe; otherwise it falls back to
# picking a single warm top, same as before layering existed.
WARDROBE_LAYER_ROLES = ["base", "mid"]
WARDROBE_LAYERING_MAX_TEMP = 40

# Free-text occasion -> target formality. Deterministic substring match;
# anything unmatched falls through to embedding similarity alone.
WARDROBE_FORMALITY_KEYWORDS = {
    "interview": 5, "wedding": 5, "formal": 5, "funeral": 5, "gala": 5,
    "presentation": 4, "office": 4, "work": 4, "meeting": 4,
    "dinner": 4, "date": 4,
    "class": 3, "school": 3, "lecture": 3, "church": 4,
    "casual": 2, "errand": 2, "grocer": 2, "hang": 2, "coffee": 2,
    "gym": 1, "workout": 1, "run": 1, "lounge": 1,
}
WARDROBE_FORMALITY_TOLERANCE = 1
WARDROBE_ALTERNATES_PER_CATEGORY = 2  # extras returned alongside each pick

# --- Text-to-speech (Kokoro via kokoro-onnx) ------------------------------
MODELS_DIR = BASE_DIR / "models"
KOKORO_MODEL_PATH = MODELS_DIR / "kokoro-v1.0.int8.onnx"
KOKORO_VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"
KOKORO_VOICE = "af_heart"
KOKORO_SPEED = 1.0
KOKORO_LANG = "en-us"

# --- FastAPI (phone-to-PC) -------------------------------------------------
# Bound to all interfaces so the Tailscale interface can reach it; the
# real access control is the X-MIDAS-Token header, not the bind address.
API_HOST = "0.0.0.0"
API_PORT = 8420
API_TOKEN = os.environ.get("MIDAS_API_TOKEN", "change-me-dev-token")
