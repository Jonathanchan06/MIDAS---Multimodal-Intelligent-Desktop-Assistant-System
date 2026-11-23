"""MIDAS's main window: a voice-reactive dot grid (ui/dot_grid.py) filling
the whole window, with the conversation drawn on top of it.

Everything sits on the grid's own canvas — status line, messages
(ui/chat_view.py), greeting, and the composer — because Tk widgets are
opaque rectangles that would cut holes in the grid. Around each piece of
UI the grid is blurred into a dim glow (ui/frost.py) so text stays
readable; bubbles and the composer box are drawn into the grid image. The
only real widget on top is the composer's text field, sized to sit inside
its drawn box.

Generation runs on a daemon thread; tokens, speaking state and voice
levels flow back through a queue.Queue drained by after()-polling, so
nothing touches Tkinter widgets off the main thread and the window never
freezes. The animation loop runs at ~30fps only while something is moving
(pointer, ripples, thinking, speech, layout changes) and idles at a slow
poll otherwise; grid frames are drawn on a background thread
(GridCanvas), so the UI thread only uploads them.

Closing the window hides it instead of quitting — the floating orb
(ui/floating_orb.py) stays on screen as the status light and the way back
in, and the scheduler + API keep running. Quit lives in the orb's
right-click menu.
"""

import ctypes
import logging
import math
import queue
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk

import config
from tools.briefing import run_morning_briefing
from ui.chat_view import ChatView
from ui.dot_grid import IDLE, SPEAKING, THINKING, GridCanvas, VoiceLevels
from ui.floating_orb import FloatingOrb
from ui.frost import Panel
from ui.theme import (
    ACCENT,
    ACCENT_HOVER,
    BG,
    COLUMN_WIDTH,
    FONT_DISPLAY_LIGHT,
    FONT_SMALL,
    FONT_TEXT,
    FONT_TEXT_SEMIBOLD,
    INK,
    LINE,
    SELECTION,
    SIZE_BODY,
    SIZE_DISPLAY,
    SIZE_SMALL,
    SLATE,
    SURFACE,
    pick_font,
)

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 12  # keep prompts small for the 2048-token chat context budget
GRID_BASE_CELL = 16  # dot spacing at 100% scaling; grows on big windows
GRID_TARGET_DOTS = 2700  # spacing widens so a maximized window stays ~this many dots
TIME_GAP_SECONDS = 10 * 60  # show a time divider after this long a pause
MAX_COMPOSER_LINES = 5
FRAME_MS = 33
IDLE_POLL_MS = 250

STATUS_TEXT = {IDLE: "Ready", THINKING: "Thinking…", SPEAKING: "Speaking"}


def _send_icon(fill: str, arrow: str, size: int) -> ImageTk.PhotoImage:
    """Round send button: a filled circle with an up arrow (round caps),
    drawn at 4x and downsampled for smooth edges."""
    px = size * 4
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((0, 0, px - 1, px - 1), fill=fill)
    w = round(px * 0.055)
    cx, top, bottom, wing = px / 2, px * 0.31, px * 0.69, px * 0.13
    for a, b in (((cx, bottom), (cx, top)), ((cx - wing, top + wing), (cx, top)), ((cx + wing, top + wing), (cx, top))):
        d.line([a, b], fill=arrow, width=w)
        for x, y in (a, b):
            d.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=arrow)
    return ImageTk.PhotoImage(img.resize((size, size), Image.LANCZOS))


def _reduced_motion() -> bool:
    """Windows' "Animation effects" setting (Settings > Accessibility >
    Visual effects)."""
    if sys.platform != "win32":
        return False
    try:
        enabled = ctypes.c_int(1)
        ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(enabled), 0)
        return not enabled.value
    except Exception:
        return False


def _colorref(hex_color: str) -> int:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return r | (g << 8) | (b << 16)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


class MidasApp(ctk.CTk):
    def __init__(self, orchestrator, tts_engine):
        super().__init__()
        self.orchestrator = orchestrator
        self.tts_engine = tts_engine
        self.history: list[dict] = []
        self.is_streaming = False
        self._queue: "queue.Queue" = queue.Queue()
        self._speaking = False
        self._conversation = False
        self._last_message_at: float | None = None
        self._entry_lines = 1
        self._entry_focused = False
        self._send_enabled = True
        self._send_hovered = False
        self._chat_regions: tuple[list, list] = ([], [])
        self._anim_job = None
        self._anim_idle = False
        self._last_tick = time.perf_counter()
        self.voice = VoiceLevels()
        self.reduced_motion = _reduced_motion()

        ctk.set_appearance_mode("dark")
        self.title("MIDAS")
        self.geometry("760x900")
        self.minsize(480, 560)
        self.configure(fg_color=BG)

        try:
            self._scale = ctk.ScalingTracker.get_window_scaling(self)
        except Exception:
            self._scale = 1.0
        s = self._scale

        # Everything is drawn on a plain-Tk canvas, which doesn't get CTk's
        # DPI scaling — sizes are physical pixels throughout.
        text = pick_font(FONT_TEXT)
        self.f_body = tkfont.Font(family=text, size=-round(SIZE_BODY * s))
        self.f_name = tkfont.Font(family=pick_font(FONT_TEXT_SEMIBOLD), size=-round(SIZE_BODY * s))
        self.f_small = tkfont.Font(family=pick_font(FONT_SMALL), size=-round(SIZE_SMALL * s))
        self.f_display = tkfont.Font(family=pick_font(FONT_DISPLAY_LIGHT), size=-round(SIZE_DISPLAY * s))

        btn = round(36 * s)
        self.icon_send = _send_icon(ACCENT, BG, btn)
        self.icon_send_hover = _send_icon(ACCENT_HOVER, BG, btn)
        self.icon_send_disabled = _send_icon(LINE, SLATE, btn)

        self.canvas = GridCanvas(
            self,
            self.voice,
            bg=BG,
            base_cell=round(GRID_BASE_CELL * s),
            on_activity=self._wake,
            frost_feather=18 * s,
        )
        self.canvas.reduced_motion = self.reduced_motion
        self.canvas.pack(fill="both", expand=True)

        self._build_status()
        self._build_greeting()
        self._build_composer()
        self.chat = ChatView(
            self.canvas,
            body_font=self.f_body,
            small_font=self.f_small,
            scale=s,
            on_regions=self._on_chat_regions,
        )

        self.mute_var = tk.BooleanVar(value=False)
        self.mute_var.trace_add(
            "write", lambda *_: setattr(self.tts_engine, "muted", self.mute_var.get())
        )
        self.floating_orb = FloatingOrb(
            self,
            voice=self.voice,
            on_toggle_window=self.toggle_window,
            on_run_briefing=self.run_briefing_now,
            on_quit=self.quit_app,
            is_window_visible=self.is_window_visible,
            mute_var=self.mute_var,
        )
        self.floating_orb.set_reduced_motion(self.reduced_motion)
        self.floating_orb.canvas.bind("<Enter>", lambda _e: self._wake(), add="+")
        self.protocol("WM_DELETE_WINDOW", self.hide_window)
        # Both fire on TTS threads — hand off through the queue, never
        # touch widgets from there.
        self.tts_engine.add_listener(lambda speaking: self._queue.put(("speaking", speaking)))
        self.tts_engine.add_level_listener(lambda frames: self._queue.put(("levels", frames)))

        self.canvas.bind("<Configure>", lambda _e: self._layout(), add="+")
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.after(50, self._style_title_bar)
        self.after_idle(self.entry.focus_set)
        self._animate()
        self._poll_queue()

    # -- building -------------------------------------------------------------

    def _build_status(self):
        c = self.canvas
        self.name_item = c.create_text(0, 0, text="MIDAS", anchor="nw", font=self.f_name, fill=INK)
        self.status_item = c.create_text(
            0, 0, text=STATUS_TEXT[IDLE], anchor="nw", font=self.f_small, fill=SLATE
        )
        self.model_item = c.create_text(
            0, 0, text=config.ORCHESTRATOR_MODEL, anchor="ne", font=self.f_small, fill=SLATE
        )

    def _build_greeting(self):
        c = self.canvas
        self.greeting_item = c.create_text(
            0, 0, text="What can I help with?", anchor="sw", font=self.f_display, fill=INK
        )
        self.privacy_item = c.create_text(
            0,
            0,
            text="Runs on this PC. Your conversations never leave it.",
            anchor="n",
            font=self.f_small,
            fill=SLATE,
        )

    def _build_composer(self):
        c = self.canvas
        self.entry = tk.Text(
            c,
            bg=SURFACE,
            fg=INK,
            font=self.f_body,
            insertbackground=INK,
            selectbackground=SELECTION,
            selectforeground=INK,
            relief="flat",
            bd=0,
            highlightthickness=0,
            wrap="word",
            undo=True,
            padx=0,
            pady=0,
        )
        self.entry_window = c.create_window(0, 0, window=self.entry, anchor="nw")
        self.entry.bind("<Return>", self._on_send)
        self.entry.bind("<Shift-Return>", self._on_newline)
        self.entry.bind("<KeyRelease>", lambda _e: self._sync_entry())
        self.entry.bind("<<Paste>>", lambda _e: self.after_idle(self._sync_entry), add="+")
        self.entry.bind("<FocusIn>", lambda _e: self._set_entry_focus(True))
        self.entry.bind("<FocusOut>", lambda _e: self._set_entry_focus(False))

        self.placeholder = tk.Label(
            c, text="Message MIDAS", font=self.f_body, fg=SLATE, bg=SURFACE, bd=0, padx=0, pady=0
        )
        self.placeholder.place(in_=self.entry, x=0, y=0)
        self.placeholder.bind("<Button-1>", lambda _e: self.entry.focus_set())

        self.send_item = c.create_image(0, 0, image=self.icon_send, anchor="center")
        c.tag_bind(self.send_item, "<Button-1>", self._on_send)
        c.tag_bind(self.send_item, "<Enter>", lambda _e: self._send_hover(True))
        c.tag_bind(self.send_item, "<Leave>", lambda _e: self._send_hover(False))

    # -- layout -----------------------------------------------------------------

    def _layout(self):
        """Position everything for the current canvas size and state, then
        hand the grid the regions to blur."""
        c, s = self.canvas, self._scale
        W, H = c.winfo_width(), c.winfo_height()
        if W < 2 or H < 2:
            return
        c.set_cell(round(math.sqrt(W * H / GRID_TARGET_DOTS)))

        col_w = min(COLUMN_WIDTH * s, W - 32 * s)
        L = (W - col_w) / 2
        R = L + col_w
        self._col = (L, R)

        # status line
        y = 18 * s
        c.coords(self.name_item, L, y)
        name_box = c.bbox(self.name_item)
        y_small = y + self.f_name.metrics("ascent") - self.f_small.metrics("ascent")
        c.coords(self.status_item, name_box[2] + 10 * s, y_small)
        c.coords(self.model_item, R, y_small)
        status_bottom = y + self.f_name.metrics("linespace")

        # composer
        line = self.f_body.metrics("linespace")
        text_h = self._entry_lines * line + 2
        pad_y = 13 * s
        comp_h = text_h + 2 * pad_y
        privacy_h = 0 if self._conversation else 30 * s
        comp_y1 = H - 20 * s - privacy_h
        comp_y0 = comp_y1 - comp_h
        btn = 36 * s
        c.coords(self.entry_window, L + 20 * s, comp_y0 + pad_y)
        c.itemconfigure(self.entry_window, width=max(20, col_w - 20 * s - 16 * s - btn), height=text_h)
        single_h = line + 2 + 2 * pad_y
        c.coords(self.send_item, R - 8 * s - btn / 2, comp_y1 - single_h / 2)
        self._composer_box = (L, comp_y0, R, comp_y1)

        # empty-state greeting and privacy line
        state = "hidden" if self._conversation else "normal"
        c.itemconfigure(self.greeting_item, state=state)
        c.itemconfigure(self.privacy_item, state=state)
        c.coords(self.greeting_item, L, comp_y0 - 24 * s)
        c.coords(self.privacy_item, (L + R) / 2, comp_y1 + 10 * s)

        self.chat.set_viewport(L, status_bottom + 18 * s, R, comp_y0 - 16 * s)
        self._push_overlay()

    def _on_chat_regions(self, halos, panels):
        self._chat_regions = (halos, panels)
        if hasattr(self, "_composer_box"):
            self._push_overlay()

    def _push_overlay(self):
        c, s = self.canvas, self._scale

        def halo_around(item, pad, r):
            box = c.bbox(item)
            return (box[0] - pad, box[1] - pad * 0.6, box[2] + pad, box[3] + pad * 0.6, r) if box else None

        halos = list(self._chat_regions[0])
        panels = list(self._chat_regions[1])
        name_box, status_box = c.bbox(self.name_item), c.bbox(self.status_item)
        if name_box and status_box:
            halos.append((name_box[0] - 14 * s, name_box[1] - 8 * s, status_box[2] + 14 * s, name_box[3] + 8 * s, 18 * s))
        halos.append(halo_around(self.model_item, 14 * s, 16 * s))
        if not self._conversation:
            halos.append(halo_around(self.greeting_item, 20 * s, 26 * s))
            halos.append(halo_around(self.privacy_item, 14 * s, 14 * s))
        L, y0, R, y1 = self._composer_box
        halos.append((L - 16 * s, y0 - 16 * s, R + 16 * s, y1 + 16 * s, 34 * s))
        panels.append(
            Panel(
                (L, y0, R, y1),
                min(24 * s, (y1 - y0) / 2),
                _rgb(SURFACE),
                1.0,
                _rgb(ACCENT if self._entry_focused else LINE),
                max(1.0, round(s)),
            )
        )
        self.canvas.set_overlay([h for h in halos if h], panels)

    def _on_wheel(self, event):
        if self.chat.contains(event.x, event.y):
            return self.chat.on_wheel(event)
        return None

    def _style_title_bar(self):
        """Windows 11: black caption so the title bar reads as part of the
        window. Ignored (harmlessly) on Windows 10."""
        if sys.platform != "win32":
            return
        try:
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            dwm = ctypes.windll.dwmapi
            for attr, value in ((20, 1), (35, _colorref(BG)), (36, _colorref(INK)), (34, _colorref(LINE))):
                v = ctypes.c_int(value)
                dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))
        except Exception:
            logger.debug("couldn't style the title bar", exc_info=True)

    # -- composer ----------------------------------------------------------------

    def _entry_value(self) -> str:
        return self.entry.get("1.0", "end-1c")

    def _sync_entry(self):
        if self._entry_value():
            self.placeholder.place_forget()
        else:
            self.placeholder.place(in_=self.entry, x=0, y=0)
        try:
            counted = self.entry.count("1.0", "end-1c", "displaylines")
        except tk.TclError:
            counted = None
        n = counted[0] if isinstance(counted, tuple) else (counted or 0)
        lines = max(1, min(MAX_COMPOSER_LINES, n + 1))
        if lines != self._entry_lines:
            self._entry_lines = lines
            self._layout()
            self.entry.see("insert")

    def _on_newline(self, _event=None):
        self.entry.insert("insert", "\n")
        self._sync_entry()
        return "break"

    def _set_entry_focus(self, focused: bool):
        self._entry_focused = focused
        self._push_overlay()

    def _set_send_enabled(self, enabled: bool):
        self._send_enabled = enabled
        self._update_send_icon()

    def _send_hover(self, inside: bool):
        self._send_hovered = inside
        self.canvas.configure(cursor="hand2" if inside and self._send_enabled else "")
        self._update_send_icon()

    def _update_send_icon(self):
        if not self._send_enabled:
            icon = self.icon_send_disabled
        elif self._send_hovered:
            icon = self.icon_send_hover
        else:
            icon = self.icon_send
        self.canvas.itemconfigure(self.send_item, image=icon)

    # -- transcript -----------------------------------------------------------

    def _enter_conversation(self):
        if not self._conversation:
            self._conversation = True
            self._layout()

    def _maybe_time_divider(self):
        now = time.time()
        if self._last_message_at is None or now - self._last_message_at > TIME_GAP_SECONDS:
            self.chat.add_time(datetime.now().strftime("%I:%M %p").lstrip("0").lower())
        self._last_message_at = now

    def _add_user_message(self, content: str):
        self._enter_conversation()
        self._maybe_time_divider()
        self.chat.add_user(content)

    def _start_assistant_turn(self):
        self.chat.start_assistant()
        self.is_streaming = True
        self._set_send_enabled(False)
        self._refresh_activity()

    # -- activity / animation ------------------------------------------------------

    def _refresh_activity(self):
        if self._speaking:
            state = SPEAKING
        elif self.is_streaming:
            state = THINKING
        else:
            state = IDLE
        self.canvas.set_state(state)
        self.floating_orb.set_state(state)
        self.canvas.itemconfigure(self.status_item, text=STATUS_TEXT[state])
        self._push_overlay()  # the status text changed width
        self._wake()

    def _wake(self):
        """Something started moving: run the next frame now, not at the
        next slow idle poll."""
        if self._anim_idle:
            self._anim_idle = False
            if self._anim_job is not None:
                self.after_cancel(self._anim_job)
            self._anim_job = self.after(1, self._animate)

    def _animate(self):
        now = time.perf_counter()
        dt = min(0.1, now - self._last_tick)
        self._last_tick = now
        self.voice.update(now, dt)
        busy = self.voice.active
        # The window's grid is skipped while hidden — no point drawing an
        # invisible canvas. The floating orb always animates.
        if self.is_window_visible():
            self.canvas.tick(dt)
            busy = busy or self.canvas.busy()
        self.floating_orb.tick(dt)
        busy = busy or self.floating_orb.busy()
        # The grid renders on its own thread and skips ticks while a frame
        # is still in flight, so a slow frame lowers its fps, not the UI's.
        delay = FRAME_MS if busy else IDLE_POLL_MS
        self._anim_idle = not busy
        self._anim_job = self.after(delay, self._animate)

    # -- window visibility --------------------------------------------------------

    def is_window_visible(self) -> bool:
        return self.state() not in ("withdrawn", "iconic")

    def hide_window(self):
        self.withdraw()
        self.floating_orb.raise_to_top()

    def show_window(self):
        self.deiconify()
        self.lift()
        # Windows won't always let a background process steal foreground;
        # a brief topmost toggle reliably brings the window forward.
        self.attributes("-topmost", True)
        self.after_idle(self.attributes, "-topmost", False)
        self.focus_force()
        self.entry.focus_set()
        self.floating_orb.raise_to_top()
        self._wake()

    def toggle_window(self):
        if self.is_window_visible():
            self.hide_window()
        else:
            self.show_window()

    def quit_app(self):
        # mainloop() returns after destroy(); main.py's finally block then
        # shuts the scheduler down. API/worker threads are daemons.
        self.tts_engine.stop()
        self.floating_orb.destroy()
        self.destroy()

    # -- briefing on demand ---------------------------------------------------------

    def run_briefing_now(self):
        """Orb-menu "Run briefing now". Goes into the transcript (and
        history) as if the user had asked for it, so follow-ups like
        "tell me more about that headline" have context. Doesn't pop the
        window open — it's spoken either way."""
        if self.is_streaming:
            return
        request = "Give me my morning briefing."
        self._add_user_message(request)
        self._start_assistant_turn()
        threading.Thread(target=self._briefing_worker, args=(request,), daemon=True).start()

    def _briefing_worker(self, request: str):
        try:
            prose = self.orchestrator.format_briefing(run_morning_briefing())
        except Exception as exc:
            logger.exception("on-demand briefing failed")
            self._queue.put(("done", (request, f"[error: {exc}]")))
            return
        self._queue.put(("done", (request, prose)))
        self.tts_engine.stop()  # an explicit request beats anything still playing
        self.tts_engine.speak(prose)

    # -- send / streaming ---------------------------------------------------

    def _on_send(self, event=None):
        text = self._entry_value().strip()
        if not text or self.is_streaming:
            return "break"
        self.entry.delete("1.0", "end")
        self._sync_entry()

        self._add_user_message(text)
        self._start_assistant_turn()

        history_snapshot = list(self.history)
        threading.Thread(
            target=self._worker, args=(text, history_snapshot), daemon=True
        ).start()
        return "break"

    def _worker(self, text: str, history_snapshot: list[dict]):
        def on_token(token: str):
            self._queue.put(("token", token))

        def on_route(model: str):
            self._queue.put(("route", model))

        # A new message cuts off whatever MIDAS is still saying — otherwise
        # replies queue behind each other and speech falls further and
        # further behind the conversation.
        self.tts_engine.stop()
        utterance = self.tts_engine.start_utterance()
        try:
            full_text = self.orchestrator.stream_response(
                text, history_snapshot, on_token, on_route, on_speech=utterance.feed
            )
        except Exception as exc:
            full_text = f"[error: {exc}]"
        finally:
            utterance.close()  # speak whatever's left; returns immediately

        self._queue.put(("done", (text, full_text)))

    def _poll_queue(self):
        tokens = []
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "token":
                    if self.is_streaming:
                        tokens.append(payload)
                    continue
                if tokens:  # keep ordering: flush tokens before other events
                    self.chat.append("".join(tokens))
                    tokens = []
                if kind == "route":
                    self.canvas.itemconfigure(self.model_item, text=payload)
                    self._push_overlay()
                elif kind == "levels":
                    self.voice.push(payload)
                    self._wake()
                elif kind == "speaking":
                    self._speaking = payload
                    if not payload:
                        self.voice.clear()
                    self._refresh_activity()
                elif kind == "done":
                    user_text, full_text = payload
                    self.chat.finish(full_text)
                    self.history.append({"role": "user", "content": user_text})
                    self.history.append({"role": "assistant", "content": full_text})
                    self.history = self.history[-HISTORY_LIMIT:]
                    self.is_streaming = False
                    self._set_send_enabled(True)
                    self._refresh_activity()
        except queue.Empty:
            pass
        if tokens:  # one re-wrap per poll, however many tokens arrived
            self.chat.append("".join(tokens))
        self.after(30, self._poll_queue)
