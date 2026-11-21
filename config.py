"""Central configuration for MIDAS. Every tunable lives here so behavior
can be adjusted without touching orchestration, tool, or UI code."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Ollama models -----------------------------------------------------
# Small, single-purpose models rather than one large model, so the 6GB
# VRAM budget only ever has to hold one of them at a time.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

ORCHESTRATOR_MODEL = "qwen3:8b"      # general chat + tool calling
# qwen3 is a hybrid reasoning model. Thinking adds seconds of hidden tokens
# before every reply/extraction and eats the small CHAT_NUM_CTX, so it's
# off for every ORCHESTRATOR_MODEL call. Only pass `think` to that model —
# the router/vision models don't support it.
ORCHESTRATOR_THINK = False
ROUTER_MODEL = "arch-router:1.5b"    # single-word traffic classifier
VISION_MODEL = "qwen2.5vl:3b"        # OCR/notification-screenshot extraction
EMBED_MODEL = "nomic-embed-text"     # 768-dim text embeddings for retrieval
# Not used anywhere until core/memory.py — must be pulled once separately:
#   ollama pull nomic-embed-text

# GPU placement. The router and embedder run on CPU (num_gpu=0 layers on
# GPU) so ORCHESTRATOR_MODEL can stay resident in VRAM. With the router on
# GPU, qwen3:8b (~4.1GB of VRAM) and arch-router (~1.1GB) couldn't
# co-reside, so EVERY message unloaded one and reloaded the other: ~4s
# router load + ~8.5s chat reload ≈ 12s before any work (measured, same
# with qwen2.5:7b). On CPU the router takes ~1.4s warm (Ollama caches the
# fixed route-list prompt prefix; the first call after a load is ~8s).
ROUTER_NUM_GPU = 0
EMBED_NUM_GPU = 0

# Keep-alives. The CPU-resident models only cost system RAM (~1.1GB
# router, ~0.3GB embedder), so they stay loaded long — reloading the
# router also throws away its prompt cache. The chat model is now the only
# thing normally in VRAM, so it can stay too; when the vision model needs
# VRAM overnight, Ollama evicts the idle chat model to make room.
ROUTER_KEEP_ALIVE = "30m"
CHAT_KEEP_ALIVE = "10m"
VISION_KEEP_ALIVE = "30s"  # used a handful of times overnight, evict fast
EMBED_KEEP_ALIVE = "30m"

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

# Sampling for every ORCHESTRATOR_MODEL call (chat, extraction, prose):
# Qwen's recommended non-thinking-mode settings. NOT low-temperature —
# qwen3 near-greedy falls into endless repetition (observed: extracting
# "what should I wear today" at temperature 0.1 looped "general advice
# for the day ahead / ..." forever). No presence_penalty: Qwen lists it
# as an anti-repetition option, but at 1.5 it caused language mixing
# (Chinese characters inside extracted English fields). Loops are instead
# made impossible in extraction by EXTRACTION_MAX_STRING_LENGTH, and
# bounded elsewhere by the *_MAX_TOKENS caps below.
# If ORCHESTRATOR_MODEL goes back to qwen2.5:7b, {"temperature": 0.3}
# is what it was tuned with.
ORCHESTRATOR_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}

# Default JSON-schema maxLength the orchestrator adds to every free-text
# string field it asks the model to fill (enum fields and fields with
# their own maxLength are left alone). Ollama enforces it in the grammar,
# so a repetition loop inside a string can't happen at all.
EXTRACTION_MAX_STRING_LENGTH = 120

# Output caps (num_predict). Backstops, not targets: Ollama doesn't stop
# a runaway generation at num_ctx — it context-shifts and keeps going, so
# without a cap one bad generation hangs the caller indefinitely.
EXTRACTION_MAX_TOKENS = 256   # real extractions are ~15-60 tokens
PROSE_MAX_TOKENS = 400        # briefing / outfit spoken prose
CHAT_MAX_TOKENS = 1024

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

# Rotation: items marked worn (mark_worn tool) within this many days are
# skipped when ranking a category, unless that would leave nothing.
WARDROBE_REWEAR_DAYS = 3
# "I'm wearing that" refers to the last suggested outfit only if it was
# suggested within this window; older than that, MIDAS asks which items.
WARDROBE_SUGGESTION_TTL_HOURS = 12

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
# fp16, not int8: measured on this laptop (Ryzen 7 7735HS, idle CPU) int8
# ran at 2.7x slower than real time vs ~0.3x for fp16 (fp32 ~0.4x) — int8
# ONNX ops are poorly optimized on many CPUs despite the smaller file.
KOKORO_MODEL_PATH = MODELS_DIR / "kokoro-v1.0.fp16.onnx"
KOKORO_VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"
KOKORO_VOICE = "af_heart"
KOKORO_SPEED = 1.0
KOKORO_LANG = "en-us"
# Longest run of text sent to synthesis without a sentence end — longer
# unpunctuated text is cut at a comma/space so speech isn't held back.
TTS_MAX_CHUNK_CHARS = 250

# --- FastAPI (phone-to-PC) -------------------------------------------------
# Bound to all interfaces so the Tailscale interface can reach it; the
# real access control is the X-MIDAS-Token header, not the bind address.
API_HOST = "0.0.0.0"
API_PORT = 8420
API_TOKEN = os.environ.get("MIDAS_API_TOKEN", "change-me-dev-token")
