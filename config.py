"""Central configuration for MIDAS. Every tunable lives here so behavior
can be adjusted without touching orchestration, tool, or UI code."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Ollama models -----------------------------------------------------
<<<<<<< HEAD
# Small, single-purpose models rather than one large model, so the 6GB
# VRAM budget only ever has to hold one of them at a time.
=======
# Two small, single-purpose models rather than one large model, so the
# 6GB VRAM budget only ever has to hold one of them at a time.
>>>>>>> b4bdb17ef4b649cf4b58fdde635b7a1ed41829c3
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

ORCHESTRATOR_MODEL = "qwen2.5:7b"    # general chat + tool calling
ROUTER_MODEL = "arch-router:1.5b"    # single-word traffic classifier
VISION_MODEL = "qwen2.5vl:3b"        # OCR/notification-screenshot extraction

# Short keep-alives so idle models evict from VRAM quickly instead of
# lingering and starving whichever model runs next.
ROUTER_KEEP_ALIVE = "30s"
CHAT_KEEP_ALIVE = "2m"
<<<<<<< HEAD
VISION_KEEP_ALIVE = "30s"  # used a handful of times overnight, evict fast
=======
>>>>>>> b4bdb17ef4b649cf4b58fdde635b7a1ed41829c3

# Aggressive context budgets appropriate for a 6GB card.
ROUTER_NUM_CTX = 1024
CHAT_NUM_CTX = 2048
<<<<<<< HEAD
# A real iPhone screenshot needs far more image tokens than a tiny test
# image — observed 3882 tokens against a real lock-screen capture, so
# budget well above that for headroom across different screen sizes.
VISION_NUM_CTX = 8192
=======
>>>>>>> b4bdb17ef4b649cf4b58fdde635b7a1ed41829c3

# Lower than Ollama's default (0.8) — small models stay on-instruction,
# classify routes more consistently, and extract structured arguments
# more reliably at low temperature.
ROUTER_TEMPERATURE = 0.1
CHAT_TEMPERATURE = 0.3

# --- Storage -------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "midas.db"

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
