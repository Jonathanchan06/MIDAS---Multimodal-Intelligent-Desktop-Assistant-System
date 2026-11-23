# Setup and guide

Everything needed to install, configure and extend MIDAS. For a short overview, see the [README](../README.md).

## Prerequisites

- **Python 3.12+** (not 3.9 — `onnxruntime`, a `kokoro-onnx` dependency,
  dropped Python 3.9 wheels, and 3.9 itself is past end-of-life). 3.13
  works too. All commands below use the project's virtual environment,
  `venv\Scripts\python.exe`, created in step 3. Using it directly
  avoids picking up the wrong Python when several are installed. (Windows'
  `py` launcher doesn't always have 3.12 registered; check with `py -0`.)
- **Ollama** installed and running.

## 1. Install Ollama models

```
ollama pull qwen3:8b
```

(`qwen3:8b` is the chat model, `ORCHESTRATOR_MODEL` in `config.py`. The
previous chat model, `qwen2.5:7b`, still works if you set it there instead:
it's faster but follows instructions less well.)

`arch-router:1.5b` isn't in Ollama's standard library — pull it from
Hugging Face instead:

```
ollama pull hf.co/katanemo/Arch-Router-1.5B.gguf
```

If that fails with `realm host "huggingface.co" does not match original
host "hf.co"`, that's a known Ollama bug
([ollama/ollama#15661](https://github.com/ollama/ollama/issues/15661)).
Work around it with the fully-qualified host:

```
ollama pull huggingface.co/katanemo/Arch-Router-1.5B.gguf
```

Either way, the pulled tag won't be named `arch-router:1.5b` — alias it
to match `config.py` (cheap, local, no re-download):

```
ollama cp hf.co/katanemo/Arch-Router-1.5B.gguf:latest arch-router:1.5b
```

Also pull the vision model used for overnight notification-screenshot
extraction (see "Overnight notification summary" below):

```
ollama pull qwen2.5vl:3b
```

And the embedding model used for wardrobe/outfit matching (see "Wardrobe
/ outfit suggestions" below):

```
ollama pull nomic-embed-text
```

Confirm all four are present:

```
ollama list
```

## 2. Get the Kokoro TTS model files

MIDAS uses `kokoro-onnx` (ONNX Runtime, CPU by default — keeps TTS off
the VRAM your Ollama models are using). Download the fp16 model and its
voice pack from the
[kokoro-onnx releases page](https://github.com/thewh1teagle/kokoro-onnx/releases)
(`model-files-v1.0`) and place them at:

```
MIDAS/models/kokoro-v1.0.fp16.onnx
MIDAS/models/voices-v1.0.bin
```

Don't use the int8 file despite it being the smallest — on this
laptop's CPU (Ryzen 7 7735HS) it synthesized at 2.7x *slower* than real
time vs ~0.3x for fp16, i.e. ~19s before the first sentence could play.

These filenames must match `KOKORO_MODEL_PATH` / `KOKORO_VOICES_PATH` in
`config.py`. If the files are missing, `TTSEngine` logs a warning and
speech becomes a no-op — the rest of the app still runs.

## 3. Install Python dependencies

From the project folder, create the virtual environment once (any
Python 3.12+), then install into it:

```
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you already have a `venv`, just run the second line. It also picks up
anything new in `requirements.txt`, e.g. Pillow, which the UI now needs.

## 4. Configure

Edit `config.py` for your watchlist tickers, news feeds, briefing time,
`USER_NAME` (used for the "Good morning, {name}" greeting), and API port.
`API_HOST` is already set to `0.0.0.0` so the server is reachable over
Tailscale, not just loopback — the real access control is the token
below, not the bind address.

Also set `WEATHER_LAT` / `WEATHER_LON` in `config.py` to your real
coordinates — they ship as NYC placeholders, used by the wardrobe
feature's weather lookup (see below).

Set a real shared-secret token for the phone API — use `setx`, not
`set`, so it persists across terminal sessions instead of only the
current one:

```
setx MIDAS_API_TOKEN your-long-random-token
```

## 5. Run

Make sure Ollama is running (`ollama serve`, or the Ollama desktop app),
then from the project folder:

```
venv\Scripts\python.exe main.py
```

This launches the desktop UI (main thread), starts the FastAPI server on
`0.0.0.0:8420` (background thread — reachable over Tailscale, see below),
and starts the APScheduler jobs
(morning briefing + reminder-due polling). Without Ollama the window
still opens, but chat, routing, and the briefing's prose formatting
won't work.

### The window

The window is black, with a dot grid filling it that shows what MIDAS is
doing:
- **Ready:** dim and still.
- **Thinking:** a band of light sweeps across.
- **Speaking:** it reacts to MIDAS's voice. Low pitches glow warm in the
  middle, high pitches show cool at the edges.

The grid also brightens under your mouse, and a click sends a ripple.
Behind the conversation and the message box it blurs so the text stays
readable.

- **Send:** Enter sends; Shift+Enter starts a new line.
- **Copy a reply:** right-click it and choose **Copy message**.
- **Reduced motion:** if Windows' **Animation effects** setting is off
  (Settings → Accessibility → Visual effects), the sweep, shimmer, and
  ripples turn off. The grid still reacts to the voice.

### Run in the background

Closing the window doesn't quit MIDAS. It hides to a small floating orb
docked to the screen edge: a round cluster of the same dots, which
reacts to MIDAS's voice too.
- **Click:** reopen the window.
- **Right-click:** Run briefing now / Mute voice / Quit.
- **Drag:** move it.

To run with no console window at all, launch with `pythonw.exe` from the
project folder:

```
venv\Scripts\pythonw.exe main.py --background
```

`--background` starts as just the orb, without opening the window. Logs
go to `data/midas.log` since there's no console. Only one copy runs at a
time — launching again just shows a "MIDAS is already running" message.
To restart, quit the running copy first: right-click the orb and choose
**Quit MIDAS**.

To start MIDAS automatically when you sign in, create a startup shortcut
(run once in PowerShell from the project folder):

```
$s = (New-Object -ComObject WScript.Shell).CreateShortcut("$([Environment]::GetFolderPath('Startup'))\MIDAS.lnk")
$s.TargetPath = "$PWD\venv\Scripts\pythonw.exe"
$s.Arguments = "main.py --background"
$s.WorkingDirectory = "$PWD"
$s.Save()
```

Delete `MIDAS.lnk` from `shell:startup` (Win+R → `shell:startup`) to undo.
Ollama has to be running too — its desktop app starts at login by default.

## Phone-to-PC API

All routes require an `X-MIDAS-Token` header matching `MIDAS_API_TOKEN`.
Reachable over Tailscale — install Tailscale on the PC and phone (same
account/tailnet), then use the PC's `100.x.x.x` Tailscale IP in place of
`127.0.0.1` below. `API_HOST = "0.0.0.0"` is already set for this; the
token is what actually gates access, not the bind address.

- `POST /chat` `{"message": "...", "history": []}` → `{"response": "..."}`
- `POST /briefing/trigger` → runs the briefing now, returns the prose. No
  PC-side speech on purpose — this endpoint's only caller is remote (a
  phone automation doing its own TTS); the scheduled PC-native briefing
  in `scheduler/jobs.py` is a separate code path and still speaks locally.
- `GET /reminders` / `POST /reminders` `{"text": "...", "remind_at": "2026-07-14T08:00:00"}`
- `POST /notification/capture` — raw image bytes as the request body (not
  multipart/`UploadFile` — a phone Shortcut can send this as a plain
  "File" request body with nothing else to configure). Responds
  immediately with `{"status": "queued"}`; the actual vision-model
  extraction and storage happen afterward via FastAPI `BackgroundTasks`,
  since inference (plus a possible cold model load) can take longer than
  a phone's request timeout, and nothing on the phone side needs to wait
  for or see the result. See "Overnight notification summary" below.
- `GET /notification/latest` → `{"response": "..."}` with the most recent
  capture's extracted text (or `"No captures yet."`). Read-only — doesn't
  mark anything as summarized, so checking it (e.g. to confirm the first
  capture of the night worked) never causes that capture to be skipped
  from the real morning recap.

The practical no-code phone client is the iOS **Shortcuts** app: a
"Get Contents of URL" action with the token as a header, POST/GET as
appropriate, and (for `/chat`-shaped JSON responses) "Get Dictionary
Value" on key `response` to pull out the text before speaking it.

## Overnight notification summary

MIDAS can't read Instagram/WhatsApp DMs directly — no personal-account
API exists for that, and iOS has no equivalent of Android's Notification
Listener permission, so no app (including Shortcuts) can read another
app's notification content directly either. The workaround: iOS
Shortcuts has a **Take Screenshot** action, and a scheduled screenshot of
a locked, dark phone's lock screen genuinely captures whatever's
accumulated in the notification stack — confirmed by direct testing.

**Phone side** (built in Shortcuts, not code): a "Capture Notification"
shortcut (Take Screenshot → Save to Photo Album, wrapped as a standalone
shortcut and invoked via "Run Shortcut" from a Personal Automation, since
Take Screenshot can't run inline in an auto-firing automation) extended
with a POST to `/notification/capture`. Duplicate the Time of Day
automation across several points in the night (e.g. every 30-60 min) —
each capture just adds to the pile, nothing needs to be seen or heard
overnight. Optionally, wire just the *first* automation of the night to
also wait ~15-20s and then check `/notification/latest` + Speak Text, as
a "did this work tonight" confirmation before you fall asleep.

**PC side**: `core/orchestrator.py`'s `extract_notification_text()` runs
the screenshot through `config.VISION_MODEL` (`qwen2.5vl:3b`) with a
deliberately open-ended prompt ("describe every notification, one per
line" — a stricter single-line-format-plus-explicit-bailout prompt was
tried first and reliably produced false "no notifications visible"
results even on screenshots with obvious, legible content; the model
itself was fine, tested independently with a generic "describe this
image" prompt). Results land in the `notification_captures` SQLite table
(`tools/notifications.py`, same fetch-then-mark-consumed pattern as
`tools/reminders.py`'s reminder alerts). Each morning, `run_morning_briefing()`
pulls everything not yet summarized, and `BRIEFING_FORMAT_PROMPT` is
explicitly told these are cumulative overnight captures that may repeat
the same notification several times (it stays on the lock screen until
dismissed) and to consolidate duplicates into one mention rather than
reading the same message back multiple times.

## Wardrobe / outfit suggestions

`suggest_fit` ("what should I wear today", "pick me an outfit for a job
interview") picks an outfit from your own saved wardrobe using current
weather (Open-Meteo, no API key needed) plus an optional occasion,
formatted into spoken prose the same way the morning briefing is.
Wardrobe items are managed entirely through chat — there's no separate
UI, and everything lives in the `wardrobe_items` SQLite table alongside
reminders.

Setup: `ollama pull nomic-embed-text` (used to semantically match
wardrobe items against a free-text occasion like "job interview") and
set your real `WEATHER_LAT` / `WEATHER_LON` in `config.py`.

### Adding items

Talk to MIDAS directly — one or several at once:

- "add my navy peacoat to my wardrobe"
- "add a black hat and a red scarf"

Only `category` and `name` are required; `color`, `warmth_level` (1-5),
`formality` (1-5), and `tags` are optional and left unknown if you don't
mention them — `suggest_fit` treats an unknown warmth/formality as
"always eligible" rather than excluding the item, so a sparse item still
gets suggested, just less precisely targeted.

To bulk-import many items at once (e.g. seeding a fresh install), write
them as JSON matching `scripts/wardrobe_seed.example.json` and run:

```
venv\Scripts\python.exe scripts/seed_wardrobe.py scripts/wardrobe_seed.example.json
```

Add `--dry-run` to validate without inserting. This isn't idempotent —
re-running it inserts duplicates.

### Editing fields

Say what changed — only the field(s) you mention are touched, everything
else on the item is left as-is:

- "set the peacoat's warmth to 5"
- "change my sweatshirt's color to blue"
- "mark my thermal shirt as a base layer"

Refer to items by name. MIDAS matches your words against what's saved
("hoodie" finds "grey hoodie"), and if that matches more than one item
or nothing at all, it changes nothing and tells you the candidates —
just say it more specifically ("the grey zip hoodie"), or use the id
shown by "what's in my closet" ("set wardrobe item 3's warmth to 5").
Changing `name`/`category`/
`color`/`tags` re-embeds the item so search stays accurate; changing only
`warmth_level`/`formality`/`layer` skips that (no need to call the
embedding model for a value that isn't part of the embedded text).

There's no bulk-edit or find-and-replace yet — for anything chat can't
reach, a direct SQLite edit is the escape hatch:

```
venv\Scripts\python.exe -c "from core.db import get_connection; c=get_connection(); c.execute('UPDATE wardrobe_items SET tags = ? WHERE id = ?', ('cotton, casual', 5)); c.commit(); c.close()"
```

### Removing items

- "remove my old peacoat"
- "delete the grey hoodie and the black jeans"
- "remove wardrobe item 5"

Same matching as edits: an ambiguous or unmatched name deletes nothing.

### Rotation

Tell MIDAS what you actually wore and suggestions rotate:

- "I'm wearing that" / "ok I'll wear that outfit" — marks the outfit it
  just suggested (within the last `WARDROBE_SUGGESTION_TTL_HOURS`, 12h by
  default)
- "I wore my grey hoodie today" — marks specific items

Anything worn within `WARDROBE_REWEAR_DAYS` (3 by default) is skipped
when picking, unless it's the only option in its category.

### Layering

`top`-category items can optionally be tagged with a layering role —
`base` (worn against skin: thermals, undershirts) or `mid` (worn over
that: sweaters, fleece) — via "mark my thermal shirt as a base layer" /
"mark my fleece as a mid layer" (this is what the `layer` field is for,
distinct from `tags`).

On a day at or below `WARDROBE_LAYERING_MAX_TEMP` (40°F by default,
`config.py`), `suggest_fit` tries to pick one `base` + one `mid` item
together instead of a single top, on top of the usual outerwear pick if
it's cold or rainy enough (`WARDROBE_OUTERWEAR_MAX_TEMP` /
`WARDROBE_RAIN_PROBABILITY_THRESHOLD`). If your wardrobe doesn't have
both roles tagged, it silently falls back to picking one warm top
instead — layering is opportunistic, never required, and never blocks a
suggestion.

## Adding a new tool

Tool invocation is a deterministic Python decision (arch-router picks a
route name, Python matches it to a registry entry) — the chat model is
never handed a list of tools and asked whether to call one. Small models
are unreliable at that judgment call in practice: during testing,
llama3.2:3b called the briefing tool on a plain "hey, how's it going"
and hallucinated a fake `wikipedia` tool call as plain text for "what's
the capital of France" — neither of those is possible once tool choice
lives in Python instead of model output.

To add a tool:

1. Write a deterministic function in `tools/<name>.py`, a paired schema
   dict (`parameters` doubles as the JSON schema for argument extraction;
   `description` becomes its arch-router route description), and — unless
   it should get an LLM prose pass like the briefing does — a
   `describe_<name>(result) -> str` formatter next to it (plain Python
   string formatting, no LLM call, no hallucination risk).
2. Import both in `main.py` and add one entry to `TOOL_REGISTRY` with a
   unique `"route"` key, plus either `"describe": describe_<name>` or
   `"speak_prose": "<style>"` (a key into `Orchestrator._prose_formatters`
   — add a new prompt + formatter method there for a genuinely new style,
   or reuse an existing one).

No other file needs to change — `Orchestrator` builds its route list and
dispatch table from `TOOL_REGISTRY` alone. If the tool takes no required
arguments (like `list_reminders`), it runs with zero LLM calls at all.

Optional: `"post_extract": fn` runs `fn(text, arguments) -> arguments`
after extraction and before the tool call, for cases where a schema field
being *required* (needed so extraction runs at all — see
`core/orchestrator.py`'s `_run_tool_route`) forces the model to produce a
value even when the user's text doesn't actually specify one, and the
model's default guess isn't trustworthy. `tools/wardrobe.py`'s
`resolve_list_category` is the example: `list_wardrobe_items`' `category`
is required, so a fully generic "what's in my wardrobe" still has to get
some category back from extraction — observed defaulting to a fixed
wrong one rather than "all". The hook checks the extracted category
actually appears in the user's own words before trusting it, falling
back to "all" otherwise.

Keep route descriptions lean — only `schema.function.description` (not
the per-field descriptions inside `parameters`) gets concatenated into
one prompt for every registered tool and sent to `arch-router` on *every*
message. This is a shared, finite budget (`ROUTER_NUM_CTX` in
`config.py`): letting one tool's top-level description balloon with
extraction-only detail (which belongs on the relevant field inside
`parameters` instead) eats into every other tool's share and can degrade
classification broadly, not just for that tool — confirmed directly: at
10 routes this hit ~1031 estimated tokens against a 1024-token budget and
silently corrupted routing across the board until `ROUTER_NUM_CTX` was
raised to 2048. Worth spot-checking route-description length (and
re-running `eval/router_eval.py`) after adding a tool, especially as the
tool count grows.
