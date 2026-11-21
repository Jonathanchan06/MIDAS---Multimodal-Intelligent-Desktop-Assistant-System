"""Floating desktop orb: a small borderless, always-on-top window docked to
a screen edge that stays up while the main window is hidden. It's the
live status light (idle / thinking / speaking) and the way back into MIDAS.

- Left-click toggles the main window.
- Drag to move; on release it snaps to the nearest left/right edge of
  whichever monitor it was dropped on (work area, so it never sits under
  the taskbar). Position persists across restarts.
- Right-click: Open/Hide MIDAS, Run briefing now, Mute voice, Quit.

Deliberately knows nothing about the orchestrator, TTS, or scheduler — it
only gets callbacks and set_state(). Per ROADMAP.md's client-server split,
this is meant to become the laptop's thin client, driven by a remote
backend instead of in-process calls, without this file changing.

Uses a plain tk.Toplevel rather than CTkToplevel: CTkToplevel's Windows
title-bar/icon tweaks fire on a delay after creation and are pointless
(at best) on an overrideredirect window.
"""

import ctypes
import json
import logging
import sys
import tkinter as tk
from ctypes import wintypes
from typing import Callable

import customtkinter as ctk

import config
from ui.orb import OrbDrawing
from ui.theme import BG

logger = logging.getLogger(__name__)

ORB_SIZE_DIP = 64  # logical size; scaled by the window's DPI factor
EDGE_MARGIN_DIP = 10
DRAG_THRESHOLD_PX = 4  # movement below this is a click, not a drag
SNAP_STEPS = 8
SNAP_STEP_MS = 15
POSITION_FILE = config.DATA_DIR / "orb_position.json"

# Pixels this exact color are punched out via -transparentcolor, so only
# the circle shows. Must never appear in the orb drawing itself.
_TRANSPARENT_KEY = "#ff00fe"

_MONITOR_DEFAULTTONEAREST = 2


class _MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


def _monitor_work_area(x: int, y: int) -> tuple[int, int, int, int] | None:
    """(left, top, right, bottom) of the work area of the monitor nearest
    the point — handles multi-monitor setups and a saved position on a
    since-unplugged monitor. Windows-only; None elsewhere or on failure.
    Coordinates are physical pixels, which matches Tk's because
    customtkinter makes the process DPI-aware."""
    if sys.platform != "win32":
        return None
    try:
        user32 = ctypes.windll.user32
        user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
        user32.MonitorFromPoint.restype = wintypes.HMONITOR
        user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(_MONITORINFO)]
        user32.GetMonitorInfoW.restype = wintypes.BOOL

        monitor = user32.MonitorFromPoint(wintypes.POINT(x, y), _MONITOR_DEFAULTTONEAREST)
        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(_MONITORINFO)
        if not monitor or not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return None
        work = info.rcWork
        return work.left, work.top, work.right, work.bottom
    except Exception:
        logger.exception("monitor lookup failed; falling back to primary screen")
        return None


class FloatingOrb(tk.Toplevel):
    def __init__(
        self,
        master,
        *,
        on_toggle_window: Callable[[], None],
        on_run_briefing: Callable[[], None],
        on_quit: Callable[[], None],
        is_window_visible: Callable[[], bool],
        mute_var: tk.BooleanVar,
    ):
        super().__init__(master)
        self.withdraw()  # stay hidden until positioned, avoids a flash at 0,0
        self._on_toggle_window = on_toggle_window
        self._is_window_visible = is_window_visible

        self.overrideredirect(True)
        self.attributes("-topmost", True)
        key = _TRANSPARENT_KEY
        try:
            self.attributes("-transparentcolor", key)
        except tk.TclError:
            key = BG  # platform without color-key transparency: show a white square
        self.configure(bg=key)

        try:
            scale = ctk.ScalingTracker.get_window_scaling(master)
        except Exception:
            scale = 1.0
        self._size = round(ORB_SIZE_DIP * scale)
        self._margin = round(EDGE_MARGIN_DIP * scale)

        self.canvas = tk.Canvas(
            self,
            width=self._size,
            height=self._size,
            bg=key,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.canvas.pack()
        self.drawing = OrbDrawing(self.canvas, self._size, disc_fill=BG)

        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Open MIDAS", command=on_toggle_window)
        self.menu.add_command(label="Run briefing now", command=on_run_briefing)
        self.menu.add_checkbutton(label="Mute voice", variable=mute_var)
        self.menu.add_separator()
        self.menu.add_command(label="Quit MIDAS", command=on_quit)

        self._press_root: tuple[int, int] | None = None
        self._press_origin = (0, 0)
        self._dragging = False
        self._snap_job = None
        # Where *we* last put the window. Read this, not winfo_x/y: those
        # lag behind a geometry() call until Tk processes it, so a fast
        # drag-release would snap from a stale position.
        self._pos = (0, 0)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Enter>", lambda _e: self.drawing.set_hover(True))
        self.canvas.bind("<Leave>", lambda _e: self.drawing.set_hover(False))

        x, y = self._load_position()
        x, y = self._snapped(x, y)
        self.geometry(f"{self._size}x{self._size}")
        self._move_to(x, y)
        self.deiconify()

    # -- public ---------------------------------------------------------------

    def set_state(self, state: str) -> None:
        self.drawing.set_state(state)

    def tick(self) -> None:
        self.drawing.tick()

    def _move_to(self, x: int, y: int) -> None:
        self._pos = (x, y)
        self.geometry(f"+{x}+{y}")

    def raise_to_top(self) -> None:
        """Windows can quietly drop an overrideredirect window's topmost
        z-order when another window activates; re-assert it."""
        self.attributes("-topmost", True)
        self.lift()

    # -- dragging / clicking ---------------------------------------------------

    def _on_press(self, event):
        if self._snap_job is not None:
            self.after_cancel(self._snap_job)
            self._snap_job = None
        self._press_root = (event.x_root, event.y_root)
        self._press_origin = self._pos
        self._dragging = False

    def _on_motion(self, event):
        if self._press_root is None:
            return
        dx = event.x_root - self._press_root[0]
        dy = event.y_root - self._press_root[1]
        if not self._dragging and abs(dx) + abs(dy) < DRAG_THRESHOLD_PX:
            return
        self._dragging = True
        self._move_to(self._press_origin[0] + dx, self._press_origin[1] + dy)

    def _on_release(self, _event):
        if self._press_root is None:
            return
        self._press_root = None
        if self._dragging:
            self._dragging = False
            target = self._snapped(*self._pos)
            self._glide_to(*target)
            self._save_position(*target)
        else:
            self._on_toggle_window()

    def _on_right_click(self, event):
        label = "Hide MIDAS" if self._is_window_visible() else "Open MIDAS"
        self.menu.entryconfigure(0, label=label)
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    # -- edge snapping --------------------------------------------------------

    def _snapped(self, x: int, y: int) -> tuple[int, int]:
        half = self._size // 2
        area = _monitor_work_area(x + half, y + half) or (
            0,
            0,
            self.winfo_screenwidth(),
            self.winfo_screenheight(),
        )
        left, top, right, bottom = area
        center_x = x + half
        if center_x - left < right - center_x:
            x = left + self._margin
        else:
            x = right - self._size - self._margin
        y = max(top + self._margin, min(y, bottom - self._size - self._margin))
        return x, y

    def _glide_to(self, target_x: int, target_y: int) -> None:
        start_x, start_y = self._pos

        def step(n: int) -> None:
            ease = 1 - (1 - n / SNAP_STEPS) ** 3  # ease-out cubic
            x = round(start_x + (target_x - start_x) * ease)
            y = round(start_y + (target_y - start_y) * ease)
            self._move_to(x, y)
            self._snap_job = self.after(SNAP_STEP_MS, step, n + 1) if n < SNAP_STEPS else None

        step(1)

    # -- persistence ------------------------------------------------------------

    def _load_position(self) -> tuple[int, int]:
        try:
            data = json.loads(POSITION_FILE.read_text(encoding="utf-8"))
            return int(data["x"]), int(data["y"])
        except FileNotFoundError:
            pass
        except Exception:
            logger.warning("ignoring unreadable %s", POSITION_FILE)
        # Default: right edge of the primary screen, a bit below center.
        return self.winfo_screenwidth(), int(self.winfo_screenheight() * 0.6)

    def _save_position(self, x: int, y: int) -> None:
        try:
            POSITION_FILE.parent.mkdir(parents=True, exist_ok=True)
            POSITION_FILE.write_text(json.dumps({"x": x, "y": y}), encoding="utf-8")
        except Exception:
            logger.warning("couldn't save orb position to %s", POSITION_FILE)
