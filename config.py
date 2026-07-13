"""Central configuration for MIDAS. Every tunable lives here so behavior
can be adjusted without touching orchestration, tool, or UI code."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Ollama models -----------------------------------------------------
# Kept as three small, single-purpose models rather than one large model
# so the 6GB VRAM budget only ever has to hold one of them at a time.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

ORCHESTRATOR_MODEL = "llama3.2:3b"   # general chat + tool calling
CODER_MODEL = "qwen2.5-coder:3b"     # code generation / explanation
ROUTER_MODEL = "arch-router:1.5b"    # single-word traffic classifier

# Short keep-alives so idle models evict from VRAM quickly instead of
# lingering and starving whichever model runs next.
ROUTER_KEEP_ALIVE = "30s"
CHAT_KEEP_ALIVE = "2m"
CODER_KEEP_ALIVE = "2m"

# Aggressive context budgets appropriate for a 6GB card running 3B models.
ROUTER_NUM_CTX = 512
CHAT_NUM_CTX = 2048
CODER_NUM_CTX = 4096

# --- Storage -------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "midas.db"

# --- Morning briefing ----------------------------------------------------
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
KOKORO_MODEL_PATH = MODELS_DIR / "kokoro-v0_19.onnx"
KOKORO_VOICES_PATH = MODELS_DIR / "voices.bin"
KOKORO_VOICE = "af_heart"
KOKORO_SPEED = 1.0
KOKORO_LANG = "en-us"

# --- FastAPI (phone-to-PC) -------------------------------------------------
# Bound to loopback only for now; swap to the Tailscale interface IP later
# without touching any route logic.
API_HOST = "127.0.0.1"
API_PORT = 8420
API_TOKEN = os.environ.get("MIDAS_API_TOKEN", "change-me-dev-token")
