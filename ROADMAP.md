# MIDAS Roadmap

Ideas, plans, and reasoning that aren't committed work yet. Nothing here is
an instruction for the current code — `CLAUDE.md` describes what's real
today. Last updated: Nov 2026.

## Vision

A Grok-like personal assistant — personality, voice, live information,
memory — that runs **completely locally**. Models run on my own hardware;
only raw data (stock quotes, RSS, later web content) comes from the
internet. No cloud model calls.

Most of what makes an assistant *feel* smart isn't raw model size: low
latency, a consistent persona and voice, live data via tools, and memory of
past conversations. Those are all buildable on current hardware. Model size
mainly limits reasoning depth.

## Hardware plan (considering, not decided)

**Current:** MSI laptop, RTX 4050 Laptop GPU, 6GB VRAM — one model resident
at a time.

**Considering:** a headless **Mac mini M6 with 32GB** unified memory as a
dedicated, always-on MIDAS server. The laptop stays my daily machine
(monitor stays on the laptop — no HDMI switching).

- Skip the 16GB base model: ~10GB usable for models after macOS, and lower
  memory bandwidth than a discrete GPU — the 8B chat model may actually run
  *slower* than on the laptop.
- 32GB lets every model stay resident (no swaps) and fits ~30B-class models.
  Prefer **MoE models** (e.g. Qwen3-30B-A3B: ~30B params, ~3B active per
  token) — they run fast despite the Mac's bandwidth limits.
- M5 Pro with 48–64GB is the better LLM machine but much pricier.
- Alternative: desktop with a used RTX 3090 (24GB) — much faster tokens/sec
  and better for fine-tuning (CUDA), but loud, power-hungry, and a worse
  always-on server.
- The biggest win is really **always-on**: scheduler, reminders, and phone
  Shortcuts work even when the laptop is closed.

### Server setup checklist (if/when it happens)

- Initial macOS setup with a display once, then run headless.
- Prevent automatic sleep; start up automatically after power failure.
- Ollama + MIDAS backend as `launchd` services (start on boot, no login).
- Wired Ethernet for lower, steadier latency.
- Remote work via VS Code Remote-SSH, plain SSH, Tailscale, and occasional
  Screen Sharing.
- **Never expose port 11434 publicly** — Ollama has no auth. LAN/Tailscale
  only.

## Target architecture: client–server split

1. **Step 1 (trivial):** run Ollama on the server with
   `OLLAMA_HOST=0.0.0.0`; point the laptop's MIDAS at it via config.
2. **Step 2 (real goal):** FastAPI, scheduler, SQLite, and Ollama all live
   on the server. The laptop runs only a thin client — the tray icon /
   floating orb.

Open questions:
- **Where TTS runs:** audio must play on the laptop. Either keep Kokoro on
  the laptop CPU, or synthesize on the server and stream audio back.
- **File access:** files live on the laptop, the brain lives on the server.
  Needs a small laptop-side file agent that executes file operations on the
  server's instructions, or a sync step.

## Hardware profiles

Support multiple hardware targets via config instead of one fixed lineup:

- `laptop-6gb`: current lineup — router 1.5B, chat 8B, vision 3B, short
  `keep_alive`s, one model resident.
- `server-32gb`: larger/MoE chat model, everything resident.

Run the eval harness and latency profiling on both and report
quality-vs-latency per profile. "The system adapts to its hardware" is a
stronger story than "it fits one fixed limit" — and the 6GB work stays
relevant as a profile, not something the upgrade erases.

## Feature ideas, ranked

### 1. Small purpose-built models instead of LLM calls
- **ONNX router** (in progress) — replaces `arch-router:1.5b`.
- **LoRA fine-tune of a tiny model (0.5B–1.5B) for argument extraction /
  function calling**: free text → `{"tool": ..., args}` JSON.
  - On 6GB: avoids loading the 8B model for tool requests (fewer swaps).
  - On 32GB: much faster than a 30B model for structured output, more
    reliable at strict formats, frees memory for a bigger chat model.
  - Generate synthetic training data; extend the eval harness to score
    extraction accuracy; report accuracy / latency / swap count before and
    after.
  - Tooling: Unsloth/PEFT on CUDA; `mlx-lm` LoRA on a Mac (slower).
- Combined resume line: "Replaced LLM calls with purpose-built small models,
  cutting latency / VRAM swaps by X%."

### 2. Real-time voice pipeline
- Wake word (openWakeWord) → streaming STT (faster-whisper) → streamed LLM
  tokens → sentence-chunked Kokoro TTS.
- One headline metric: **time from end of speech to first audio**. Target
  < ~1s.
- A client–server split adds a network hop, which makes this harder and
  more interesting.
- This is what makes the always-on tray icon / orb actually useful (pair
  with a global hotkey).

### 3. GPU request scheduling
- Replaces the earlier "predictive model preloading" idea, which mostly
  disappears once everything stays resident on 32GB (still relevant for the
  `laptop-6gb` profile).
- When voice, background agents, and scheduled jobs hit Ollama at once:
  priority queue — interactive voice first, background jobs last. Measure
  queueing delay under load.

### 4. Memory + local RAG + safe file management
- Embed notes/documents into `sqlite-vec` (fits the existing SQLite stack);
  evaluate retrieval quality.
- Conversation memory for the "feels personal" part of the Grok vision.
- File actions with a dry-run preview and an undo log before anything
  moves.
- Depends on the laptop-side file agent (see client–server open questions).

### 5. ESP32 voice satellite
- ESP32 mic + speaker streaming audio to MIDAS over Wi-Fi — talk to it from
  another room.
- Even better with an always-on server.
- Doubles as a real test of whether I enjoy hardware/firmware work.

### Also on the list
- Personality LoRA (README's future-additions item).
- Sanitization/detection pass on OCR'd notification text (prompt-injection
  surface).
- Pre-commit hook rejecting `<<<<<<<` / `=======` / `>>>>>>>` in staged
  files.
- Packaging as a standalone `.exe`.
- Evaluate Qwen3.5 multimodal `9b` as a combined chat + vision model.

## Interview framing

- Built under a real constraint (6GB VRAM), measured everything, made
  explicit tradeoffs.
- Then evolved it: laptop-only → client–server with hardware profiles.
- Impressiveness comes from measurement and tradeoffs, not from how small
  (or big) the hardware is.
