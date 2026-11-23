"""Floating desktop orb: a small borderless, always-on-top window docked to
a screen edge that stays up while the main window is hidden. It's the
live status light (idle / thinking / speaking, voice-reactive) and the way
back into MIDAS. It draws a round cluster of the main window's dot grid on
a black disc.

- Left-click toggles the main window.
- Drag to move; on release it snaps to the nearest left/right edge of
  whichever monitor it was dropped on (work area, so it never sits under
  the taskbar). Position persists across restarts.
- Right-click: Open/Hide MIDAS, Run briefing now, Mute voice, Quit.

Deliberately knows nothing about the orchestrator, TTS, or scheduler — it
only gets callbacks, set_state(), and the UI's VoiceLevels. Per
ROADMAP.md's client-server split, this is meant to become the laptop's
thin client, driven by a remote backend instead of in-process calls.

Drawing: on Windows the window is a per-pixel-alpha layered window
(UpdateLayeredWindow), so the disc has smooth edges and a soft shadow over
any wallpaper. Tk's own painting of the window is ignored in that mode;
the canvas only exists to receive mouse events. If that setup fails (or
off Windows), it falls back to Tk's -transparentcolor colour key, whose
key is one shade off the disc's rim so the hard edge doesn't fringe.

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
import numpy as np
from PIL import Image, ImageFilter, ImageTk

import config
from ui.dot_grid import IDLE, DotField, DotRenderer, VoiceLevels
from ui.theme import LINE

logger = logging.getLogger(__name__)

ORB_SIZE_DIP = 64  # logical size; scaled by the window's DPI factor
EDGE_MARGIN_DIP = 10
DRAG_THRESHOLD_PX = 4  # movement below this is a click, not a drag
SNAP_STEPS = 8
SNAP_STEP_MS = 15
POSITION_FILE = config.DATA_DIR / "orb_position.json"

_RIM_RGB = np.array([int(LINE[i : i + 2], 16) for i in (1, 3, 5)], np.float32)
# Colour-key fallback only: pixels exactly this colour are punched out. One
# shade off the rim, so anti-aliased edge pixels blend toward the rim.
_TRANSPARENT_KEY = "#24282c"

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


# -- per-pixel alpha (Windows layered window) ---------------------------------

_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x00080000
_ULW_ALPHA = 0x2
_AC_SRC_OVER, _AC_SRC_ALPHA = 0x0, 0x1


class _BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", ctypes.c_ubyte),
        ("BlendFlags", ctypes.c_ubyte),
        ("SourceConstantAlpha", ctypes.c_ubyte),
        ("AlphaFormat", ctypes.c_ubyte),
    ]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _LayeredSurface:
    """Owns a 32-bit DIB and pushes it to a window with UpdateLayeredWindow."""

    def __init__(self, hwnd: int, size: int):
        u32 = ctypes.WinDLL("user32", use_last_error=True)
        g32 = ctypes.WinDLL("gdi32", use_last_error=True)
        u32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        u32.GetWindowLongW.restype = wintypes.LONG
        u32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
        u32.SetWindowLongW.restype = wintypes.LONG
        u32.GetDC.argtypes = [wintypes.HWND]
        u32.GetDC.restype = wintypes.HDC
        u32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        u32.UpdateLayeredWindow.argtypes = [
            wintypes.HWND, wintypes.HDC, ctypes.POINTER(wintypes.POINT),
            ctypes.POINTER(wintypes.SIZE), wintypes.HDC, ctypes.POINTER(wintypes.POINT),
            wintypes.DWORD, ctypes.POINTER(_BLENDFUNCTION), wintypes.DWORD,
        ]
        u32.UpdateLayeredWindow.restype = wintypes.BOOL
        g32.CreateCompatibleDC.argtypes = [wintypes.HDC]
        g32.CreateCompatibleDC.restype = wintypes.HDC
        g32.CreateDIBSection.argtypes = [
            wintypes.HDC, ctypes.POINTER(_BITMAPINFOHEADER), wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD,
        ]
        g32.CreateDIBSection.restype = wintypes.HBITMAP
        g32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        g32.SelectObject.restype = wintypes.HGDIOBJ
        g32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        g32.DeleteDC.argtypes = [wintypes.HDC]
        self._u32, self._g32 = u32, g32
        self._hwnd, self._size = hwnd, size

        style = u32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
        u32.SetWindowLongW(hwnd, _GWL_EXSTYLE, style | _WS_EX_LAYERED)

        screen = u32.GetDC(None)
        self._dc = g32.CreateCompatibleDC(screen)
        u32.ReleaseDC(None, screen)
        header = _BITMAPINFOHEADER(
            biSize=ctypes.sizeof(_BITMAPINFOHEADER), biWidth=size, biHeight=-size,  # top-down
            biPlanes=1, biBitCount=32, biCompression=0,
        )
        self._bits = ctypes.c_void_p()
        self._bmp = g32.CreateDIBSection(self._dc, ctypes.byref(header), 0, ctypes.byref(self._bits), None, 0)
        if not self._bmp or not self._bits:
            raise OSError("CreateDIBSection failed")
        self._old = g32.SelectObject(self._dc, self._bmp)
        self._blend = _BLENDFUNCTION(_AC_SRC_OVER, 0, 255, _AC_SRC_ALPHA)
        self._sz = wintypes.SIZE(size, size)
        self._src = wintypes.POINT(0, 0)

    def paint(self, bgra_premultiplied: np.ndarray) -> None:
        data = np.ascontiguousarray(bgra_premultiplied, dtype=np.uint8)
        ctypes.memmove(self._bits, data.ctypes.data, data.nbytes)
        ok = self._u32.UpdateLayeredWindow(
            self._hwnd, None, None, ctypes.byref(self._sz), self._dc,
            ctypes.byref(self._src), 0, ctypes.byref(self._blend), _ULW_ALPHA,
        )
        if not ok:
            raise OSError(f"UpdateLayeredWindow failed ({ctypes.get_last_error()})")

    def close(self) -> None:
        try:
            self._g32.SelectObject(self._dc, self._old)
            self._g32.DeleteObject(self._bmp)
            self._g32.DeleteDC(self._dc)
        except Exception:
            pass


# -- the orb's artwork ----------------------------------------------------------


class _OrbArt:
    """Static disc + shadow, with a circular DotField drawn on top each frame."""

    def __init__(self, size: int):
        s = size
        pad = max(3, round(s * 0.08))
        radius = s / 2 - pad
        yy, xx = np.mgrid[0:s, 0:s] + 0.5
        d = np.hypot(xx - s / 2, yy - s / 2)
        disc = np.clip(radius - d + 0.5, 0, 1).astype(np.float32)
        rim = (np.clip(1 - np.abs(d - (radius - 0.75)), 0, 1) * disc).astype(np.float32)

        # soft drop shadow: the disc, nudged down, blurred
        offset = max(1, round(s * 0.03))
        shadow_src = np.zeros((s, s), np.uint8)
        shadow_src[offset:, :] = (disc[: s - offset, :] * 255).astype(np.uint8)
        blurred = Image.fromarray(shadow_src, "L").filter(ImageFilter.GaussianBlur(max(1.5, s * 0.05)))
        shadow = np.asarray(blurred, np.float32) / 255 * 0.55

        self.size = s
        self.disc = disc
        self.alpha = disc + shadow * (1 - disc)  # shadow is black: adds alpha, no colour
        self.base_rgb = rim[..., None] * _RIM_RGB  # premultiplied (disc interior is black)

        side = int(2 * (radius - 1.5))
        self.offset = (s - side) // 2
        self.field = DotField(side, side, max(5, round(s / 9)), circle=True)
        self.field.pointer_sigma = side * 0.6
        self.field.pointer_gain = 0.45
        self.renderer = DotRenderer(self.field)

    def compose(self) -> tuple[np.ndarray, np.ndarray]:
        """(premultiplied RGB float, alpha 0..1)."""
        rgb = self.base_rgb.copy()
        o, side = self.offset, self.field.width
        rgb[o : o + side, o : o + side] += self.renderer.buffer
        np.clip(rgb, 0, 255, out=rgb)
        return rgb, self.alpha


class FloatingOrb(tk.Toplevel):
    def __init__(
        self,
        master,
        *,
        voice: VoiceLevels,
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
        self._voice = voice

        self.overrideredirect(True)
        self.attributes("-topmost", True)

        try:
            scale = ctk.ScalingTracker.get_window_scaling(master)
        except Exception:
            scale = 1.0
        self._size = round(ORB_SIZE_DIP * scale)
        self._margin = round(EDGE_MARGIN_DIP * scale)
        self._art = _OrbArt(self._size)
        self._art.field.set_state(IDLE)
        self._needs_frame = True

        self.configure(bg="#000000")
        self.canvas = tk.Canvas(
            self,
            width=self._size,
            height=self._size,
            bg="#000000",
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.canvas.pack()
        self.geometry(f"{self._size}x{self._size}")

        self._surface: _LayeredSurface | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._setup_surface()

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
        self.canvas.bind("<Enter>", lambda _e: self.set_hover(True))
        self.canvas.bind("<Leave>", lambda _e: self.set_hover(False))

        x, y = self._load_position()
        x, y = self._snapped(x, y)
        self._move_to(x, y)
        self.tick(0.0)  # first frame before the window appears
        self.deiconify()

    # -- public ---------------------------------------------------------------

    def set_state(self, state: str) -> None:
        self._art.field.set_state(state)

    def set_hover(self, hover: bool) -> None:
        c = self._art.field.width / 2
        self._art.field.set_pointer(c if hover else None, c)

    def set_reduced_motion(self, reduced: bool) -> None:
        self._art.field.reduced_motion = reduced

    def busy(self) -> bool:
        return self._needs_frame or self._art.field.busy(self._voice)

    def tick(self, dt: float) -> None:
        if not self.busy():
            return
        art = self._art
        changed = art.renderer.render(*art.field.step(dt, self._voice))
        if changed or self._needs_frame:
            self._paint()
        self._needs_frame = False

    def raise_to_top(self) -> None:
        """Windows can quietly drop an overrideredirect window's topmost
        z-order when another window activates; re-assert it."""
        self.attributes("-topmost", True)
        self.lift()

    def destroy(self) -> None:
        if self._surface is not None:
            self._surface.close()
            self._surface = None
        super().destroy()

    # -- drawing ----------------------------------------------------------------

    def _setup_surface(self) -> None:
        if sys.platform == "win32":
            try:
                self.update_idletasks()
                hwnd = ctypes.windll.user32.GetParent(self.winfo_id()) or self.winfo_id()
                self._surface = _LayeredSurface(hwnd, self._size)
                return
            except Exception:
                logger.exception("layered orb window unavailable; using colour-key transparency")
                self._surface = None
        try:
            self.attributes("-transparentcolor", _TRANSPARENT_KEY)
            self.configure(bg=_TRANSPARENT_KEY)
            self.canvas.configure(bg=_TRANSPARENT_KEY)
        except tk.TclError:
            pass  # no colour keying: the disc shows on a dark square
        self._photo = ImageTk.PhotoImage("RGB", (self._size, self._size))
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _paint(self) -> None:
        rgb, alpha = self._art.compose()
        if self._surface is not None:
            bgra = np.empty((self._size, self._size, 4), np.uint8)
            bgra[..., 0] = rgb[..., 2]
            bgra[..., 1] = rgb[..., 1]
            bgra[..., 2] = rgb[..., 0]
            bgra[..., 3] = np.rint(alpha * 255)
            try:
                self._surface.paint(bgra)
                return
            except OSError:
                logger.exception("layered paint failed; switching to colour-key transparency")
                self._surface.close()
                self._surface = None
                self._setup_surface()
        key = np.array([int(_TRANSPARENT_KEY[i : i + 2], 16) for i in (1, 3, 5)], np.float32)
        opaque = (self._art.disc >= 0.5)[..., None]
        disc = np.maximum(self._art.disc, 1e-3)[..., None]
        out = np.where(opaque, rgb / disc, key)  # un-premultiply inside the disc
        self._photo.paste(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB"))

    def _move_to(self, x: int, y: int) -> None:
        self._pos = (x, y)
        self.geometry(f"+{x}+{y}")

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
