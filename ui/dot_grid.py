"""The voice-reactive dot grid: MIDAS's presence in the main window, and (as
a small round cluster) in the floating desktop orb.

Three layers, so the drawing never knows about Tk until the last one:

- VoiceLevels: timed level frames from the TTS playback thread (handed
  over through the app's queue), released when their audio is actually
  audible, then smoothed (fast attack, slow release).
- DotField: per-dot size/brightness for one frame, from the activity state
  (idle / thinking / speaking), the mouse, click ripples and voice levels.
  Pure numpy.
- DotRenderer: draws the field into an RGB buffer. Every dot sits in its
  own grid cell, so a frame is a gather of pre-rendered anti-aliased dot
  sprites — one vectorised numpy pass, no per-dot Python loop, no
  supersampling. Only the bounding box of cells that changed since the
  last frame is redrawn.

Measured (Ryzen 7 7735HS, 760x720 window, 2115 dots): ~3.5 ms per full
frame, ~10 ms while Kokoro is synthesising on the other cores. Per-dot
ImageDraw ellipses at 2x were ~14 ms; per-dot sprite pastes ~6.5 ms.

Colour is pitch: a dot's distance from the centre picks both its colour
and the frequency band it reacts to — low bands (warm) at the centre,
high bands (cool) at the edges.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import tkinter as tk
from collections import deque

import numpy as np
from PIL import Image, ImageTk

from ui.frost import Frost, Panel

logger = logging.getLogger(__name__)

IDLE, THINKING, SPEAKING = "idle", "thinking", "speaking"

N_BANDS = 8  # must match core.tts.VOICE_BANDS
SPRITE_LEVELS = 32  # radius steps between the smallest and largest dot

# Colour stops from centre (lowest band) to edge (highest band).
_STOPS = np.array(
    [[255, 181, 71], [255, 94, 91], [169, 112, 255], [34, 211, 238]], dtype=np.float32
)

_IDLE_LEVEL = 0.14
_MIN_BRIGHTNESS = 0.30  # idle dots still show their colour on black
# The main window's grid sits behind text, so it's muted: colours pulled
# toward grey and capped below full brightness. The floating orb is a small
# status light and keeps the full palette (DotField defaults).
GRID_SATURATION = 0.55
GRID_PEAK_BRIGHTNESS = 0.70
_FRAME_S = 1 / 30


def _gradient(p: np.ndarray) -> np.ndarray:
    x = np.clip(p, 0.0, 1.0) * (len(_STOPS) - 1)
    i = np.minimum(x.astype(np.intp), len(_STOPS) - 2)
    f = (x - i)[..., None]
    return _STOPS[i] * (1 - f) + _STOPS[i + 1] * f


def _ease(rate_per_frame: float, dt: float) -> float:
    """Frame-rate-independent smoothing factor."""
    return 1.0 - (1.0 - rate_per_frame) ** (dt / _FRAME_S)


class VoiceLevels:
    """Smoothed loudness + per-band levels of what MIDAS is saying right now."""

    ATTACK = 0.6
    RELEASE = 0.14
    _HOLD_S = 0.12  # no new frame for this long -> treat as silence

    def __init__(self):
        self._pending: deque = deque()
        self._target_bands = np.zeros(N_BANDS, np.float32)
        self._target_loud = 0.0
        self._last_frame_at = 0.0
        self.bands = np.zeros(N_BANDS, np.float32)
        self.loud = 0.0

    def push(self, frames) -> None:
        """frames: iterable of (due perf_counter time, loudness 0..1, bands[8] 0..1)."""
        self._pending.extend(frames)

    def clear(self) -> None:
        self._pending.clear()
        self._target_bands[:] = 0
        self._target_loud = 0.0

    @property
    def active(self) -> bool:
        return bool(self._pending) or self.loud > 0.005 or float(self.bands.max()) > 0.005

    def update(self, now: float, dt: float) -> None:
        while self._pending and self._pending[0][0] <= now:
            _, loud, bands = self._pending.popleft()
            self._target_loud = loud
            self._target_bands[:] = bands
            self._last_frame_at = now
        if now - self._last_frame_at > self._HOLD_S:
            self._target_loud = 0.0
            self._target_bands[:] = 0
        up, down = _ease(self.ATTACK, dt), _ease(self.RELEASE, dt)
        k = np.where(self._target_bands > self.bands, up, down)
        self.bands += (self._target_bands - self.bands) * k
        self.loud += (self._target_loud - self.loud) * (up if self._target_loud > self.loud else down)
        if not self.active:
            self.bands[:] = 0
            self.loud = 0.0


class DotField:
    """Dot positions plus everything that moves them, for one surface."""

    def __init__(
        self,
        width: int,
        height: int,
        cell: int,
        circle: bool = False,
        saturation: float = 1.0,
        peak_brightness: float = 1.0,
    ):
        self.width, self.height, self.cell = width, height, cell
        self.cols = max(1, width // cell)
        self.rows = max(1, height // cell)
        self.ox = (width - self.cols * cell) // 2
        self.oy = (height - self.rows * cell) // 2
        xs = self.ox + (np.arange(self.cols) + 0.5) * cell
        ys = self.oy + (np.arange(self.rows) + 0.5) * cell
        self.px, self.py = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
        cx, cy = width / 2, height / 2
        r = np.hypot(self.px - cx, self.py - cy)
        if circle:
            radius = min(cx, cy)
            self.mask = r <= radius - cell * 0.55
            dist = r / max(1.0, radius - cell * 0.55)
        else:
            self.mask = None
            dist = r / max(1.0, math.hypot(cx, cy))
        rgb = _gradient(dist)
        grey = (rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
        self.base_rgb = (grey + (rgb - grey) * saturation).astype(np.float32)
        self._peak = peak_brightness
        # continuous band coordinate per dot, and the two bands it blends
        pos = np.clip(dist, 0, 1) * (N_BANDS - 1)
        self._band_lo = np.minimum(pos.astype(np.intp), N_BANDS - 2)
        self._band_f = (pos - self._band_lo).astype(np.float32)
        # diagonal sweep coordinate for "thinking"
        span = width + 0.4 * height
        self._sweep_u = ((self.px + 0.4 * self.py) / span).astype(np.float32)
        self._phase = np.random.default_rng(7).uniform(0, 2 * np.pi, self.px.shape).astype(np.float32)

        self.reduced_motion = False
        self.state = IDLE
        self._think = 0.0  # eased 0..1 presence of the thinking sweep
        self._pointer: tuple[float, float] | None = None
        self._pointer_xy = (cx, cy)
        self._pointer_amt = 0.0
        self.pointer_sigma = 4.5 * cell
        self.pointer_gain = 0.6
        self._ripples: list[list[float]] = []  # [x, y, age]
        self._ripple_lock = threading.Lock()
        self._t = 0.0

    # -- inputs -------------------------------------------------------------

    def set_state(self, state: str) -> None:
        self.state = state

    def set_pointer(self, x: float | None, y: float | None = None) -> None:
        if x is None:
            self._pointer = None
        else:
            self._pointer = (x, y)
            self._pointer_xy = (x, y)

    def add_ripple(self, x: float, y: float) -> None:
        if not self.reduced_motion:
            with self._ripple_lock:
                self._ripples.append([x, y, 0.0])

    def busy(self, voice: VoiceLevels) -> bool:
        return (
            self._pointer is not None
            or self._pointer_amt > 0.002
            or bool(self._ripples)
            or self.state == THINKING
            or self._think > 0.002
            or voice.active
        )

    # -- one frame ------------------------------------------------------------

    def step(self, dt: float, voice: VoiceLevels) -> tuple[np.ndarray, np.ndarray]:
        """Returns (sprite index per dot, uint16 RGB per dot)."""
        self._t += dt
        cell = self.cell
        lvl = np.full(self.px.shape, _IDLE_LEVEL, np.float32)

        target = 1.0 if self._pointer is not None else 0.0
        self._pointer_amt += (target - self._pointer_amt) * _ease(0.25, dt)
        if self._pointer_amt > 0.002:
            x, y = self._pointer_xy
            d2 = (self.px - x) ** 2 + (self.py - y) ** 2
            lvl += self.pointer_gain * self._pointer_amt * np.exp(-d2 / (2 * self.pointer_sigma**2))

        with self._ripple_lock:
            ripples = list(self._ripples)
        if ripples:
            speed, life, width = 14 * cell, 1.3, 1.4 * cell
            alive = []
            for rp in ripples:
                rp[2] += dt
                if rp[2] < life:
                    alive.append(rp)
                    ring = np.hypot(self.px - rp[0], self.py - rp[1]) - rp[2] * speed
                    lvl += 0.7 * (1 - rp[2] / life) ** 2 * np.exp(-(ring / width) ** 2)
            with self._ripple_lock:  # keep any added meanwhile
                self._ripples = alive + [r for r in self._ripples if not any(r is q for q in ripples)]

        target = 1.0 if self.state == THINKING else 0.0
        self._think += (target - self._think) * _ease(0.12, dt)
        if self._think > 0.002:
            if self.reduced_motion:
                lvl += 0.22 * self._think
            else:
                head = (self._t * 0.45) % 1.4 - 0.2
                lvl += self._think * (0.05 + 0.5 * np.exp(-(((self._sweep_u - head) / 0.07) ** 2)))

        if voice.active:
            b = voice.bands
            energy = b[self._band_lo] * (1 - self._band_f) + b[self._band_lo + 1] * self._band_f
            shimmer = 1.0 if self.reduced_motion else 0.85 + 0.15 * np.sin(self._phase + self._t * 9)
            lvl += (0.15 * voice.loud + 0.8 * energy) * shimmer

        np.clip(lvl, 0.0, 1.0, out=lvl)
        idx = np.rint(lvl**0.85 * (SPRITE_LEVELS - 1)).astype(np.intp)
        bright = self._peak * (_MIN_BRIGHTNESS + (1 - _MIN_BRIGHTNESS) * lvl)
        col = (self.base_rgb * bright[..., None]).astype(np.uint16)
        if self.mask is not None:
            idx[~self.mask] = 0
            col[~self.mask] = 0
        return idx, col


def _sprites(cell: int) -> np.ndarray:
    """(SPRITE_LEVELS, cell, cell) anti-aliased disc coverage, 0..255 (uint16)."""
    yy, xx = np.mgrid[0:cell, 0:cell] + 0.5
    dist = np.hypot(xx - cell / 2, yy - cell / 2)
    r_min, r_max = max(0.9, 0.11 * cell), 0.44 * cell
    radii = r_min + (r_max - r_min) * np.arange(SPRITE_LEVELS) / (SPRITE_LEVELS - 1)
    cov = np.clip(radii[:, None, None] - dist[None] + 0.5, 0, 1)
    return np.rint(cov * 255).astype(np.uint16)


class DotRenderer:
    """Draws a DotField's output into an RGB buffer, redrawing only changed cells."""

    def __init__(self, field: DotField):
        f = field
        self._spr = _sprites(f.cell)
        self.buffer = np.zeros((f.height, f.width, 3), np.uint8)
        gh, gw = f.rows * f.cell, f.cols * f.cell
        self._view = self.buffer[f.oy : f.oy + gh, f.ox : f.ox + gw].reshape(
            f.rows, f.cell, f.cols, f.cell, 3
        )
        self._prev_idx: np.ndarray | None = None
        self._prev_col: np.ndarray | None = None
        # fraction of a cell each sprite level covers -> per-cell average colour
        self._frac = (self._spr.reshape(SPRITE_LEVELS, -1).mean(axis=1) / 255).astype(np.float32)

    def wash(self, idx: np.ndarray, col: np.ndarray) -> np.ndarray:
        """Average colour of each cell, (rows, cols, 3) float — the dots as
        they'd look completely out of focus. Used by ui/frost.py."""
        return col.astype(np.float32) * self._frac[idx][..., None]

    def render(self, idx: np.ndarray, col: np.ndarray) -> bool:
        """Returns False when nothing changed (no upload needed)."""
        if self._prev_idx is None:
            r0, r1, c0, c1 = 0, idx.shape[0], 0, idx.shape[1]
        else:
            diff = (idx != self._prev_idx) | (
                np.abs(col.astype(np.int16) - self._prev_col.astype(np.int16)).max(axis=2) > 1
            )
            if not diff.any():
                return False
            rows = np.flatnonzero(diff.any(axis=1))
            cols = np.flatnonzero(diff.any(axis=0))
            r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        cov = self._spr[idx[r0:r1, c0:c1]]  # R, C, cell, cell
        cs = col[r0:r1, c0:c1]
        sub = self._view[r0:r1, :, c0:c1, :]  # R, cell, C, cell, 3
        for ch in range(3):
            v = cov * cs[:, :, ch, None, None]
            v >>= 8
            sub[..., ch] = v.transpose(0, 2, 1, 3)
        self._prev_idx, self._prev_col = idx.copy(), col.copy()
        return True


class _Job:
    """One frame's inputs, handed to the render thread."""

    __slots__ = ("gen", "field", "renderer", "frost", "regions", "dt", "force")

    def __init__(self, gen, field, renderer, frost, regions, dt, force):
        self.gen, self.field, self.renderer, self.frost = gen, field, renderer, frost
        self.regions, self.dt, self.force = regions, dt, force


def _render_frame(job: _Job, voice: VoiceLevels) -> Image.Image | None:
    if job.regions is not None:
        job.frost.set_regions(*job.regions)
    idx, col = job.field.step(job.dt, voice)
    if not (job.renderer.render(idx, col) or job.force):
        return None
    if not job.frost.empty:
        return job.frost.compose(job.renderer.buffer, job.renderer.wash(idx, col), job.field)
    return Image.fromarray(job.renderer.buffer, "RGB").copy()


class GridCanvas(tk.Canvas):
    """The main window's dot grid, filling the canvas behind everything else
    drawn on it. The owner calls set_state()/set_overlay()/tick(); this
    widget never schedules anything itself.

    Frames are rendered on a background thread: numpy and Pillow release
    the GIL for their heavy work, so drawing uses another core and the UI
    thread only uploads finished frames (PhotoImage.paste must stay on the
    Tk thread). A frame is presented one tick after it's requested.
    """

    def __init__(
        self, master, voice: VoiceLevels, bg: str, base_cell: int, on_activity=None, frost_feather: float = 18
    ):
        super().__init__(master, bg=bg, highlightthickness=0, bd=0)
        self.voice = voice
        self._on_activity = on_activity or (lambda: None)
        self.base_cell = base_cell
        self.cell = base_cell
        self.reduced_motion = False
        self._state = IDLE
        self._field: DotField | None = None
        self._renderer: DotRenderer | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._item = self.create_image(0, 0, anchor="nw")
        self.tag_lower(self._item)  # everything else on the canvas draws over the grid
        self._frost_feather = frost_feather
        self._frost: Frost | None = None
        self._regions: tuple | None = None  # pending (halos, panels) for the next job
        self._last_regions: tuple | None = None  # re-applied after a resize
        self._needs_frame = False
        self.render_ms = 0.0  # EMA of the render thread's frame cost

        self._gen = 0  # bumped on rebuild; frames from an older size are dropped
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._job: _Job | None = None
        self._in_flight = False
        self._ready: tuple | None = None  # (gen, frame)
        self._pending_dt = 0.0
        threading.Thread(target=self._render_loop, name="grid-render", daemon=True).start()

        self.bind("<Configure>", self._on_configure)
        self.bind("<Motion>", lambda e: self._pointer(e.x, e.y))
        self.bind("<Leave>", lambda _e: self._pointer(None))
        self.bind("<ButtonPress-1>", self._on_click)

    def set_cell(self, cell: int) -> None:
        cell = max(self.base_cell, cell)
        if cell != self.cell:
            self.cell = cell
            self._rebuild(self.winfo_width(), self.winfo_height())

    def set_state(self, state: str) -> None:
        self._state = state
        if self._field is not None:
            self._field.set_state(state)

    def set_overlay(self, halos: list, panels: list[Panel]) -> None:
        """Regions of UI drawn over the grid: halos (blurred surroundings)
        and panels (bubbles, composer). Applied with the next frame, so
        several changes in one tick cost one rebuild."""
        self._regions = self._last_regions = (halos, panels)
        self._needs_frame = True
        self._on_activity()

    def busy(self) -> bool:
        return (
            self._needs_frame
            or self._in_flight
            or self._ready is not None
            or (self._field is not None and self._field.busy(self.voice))
        )

    def tick(self, dt: float) -> None:
        """UI thread: present a finished frame, then request the next."""
        with self._lock:
            ready, self._ready = self._ready, None
        if ready is not None and ready[0] == self._gen:
            self._photo.paste(ready[1])
        if self._field is None:
            return
        self._pending_dt += dt
        if not self.busy():
            self._pending_dt = 0.0
            return
        with self._lock:
            if self._in_flight:
                return  # still drawing the last one; its time carries over
            self._job = _Job(
                self._gen, self._field, self._renderer, self._frost,
                self._regions, self._pending_dt, self._needs_frame,
            )
            self._in_flight = True
        self._regions = None
        self._pending_dt = 0.0
        self._needs_frame = False
        self._wakeup.set()

    # -- internals -------------------------------------------------------------

    def _render_loop(self) -> None:
        while True:
            self._wakeup.wait()
            self._wakeup.clear()
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            start = time.perf_counter()
            try:
                frame = _render_frame(job, self.voice)
            except Exception:
                logger.exception("dot grid frame failed")
                frame = None
            ms = (time.perf_counter() - start) * 1000
            with self._lock:
                if frame is not None and job.gen == self._gen:
                    self._ready = (job.gen, frame)
                self._in_flight = False
                self.render_ms = ms if self.render_ms == 0 else self.render_ms * 0.9 + ms * 0.1

    def _on_configure(self, event) -> None:
        if self._field is None or (event.width, event.height) != (self._field.width, self._field.height):
            self._rebuild(event.width, event.height)

    def _rebuild(self, width: int, height: int) -> None:
        if width < 2 or height < 2:
            return
        old = self._field
        field = DotField(
            width, height, self.cell, saturation=GRID_SATURATION, peak_brightness=GRID_PEAK_BRIGHTNESS
        )
        field.reduced_motion = self.reduced_motion
        field.set_state(self._state)
        if old is not None:  # keep motion continuous across resizes
            field._t, field._think = old._t, old._think
            field._pointer, field._pointer_xy = old._pointer, old._pointer_xy
        renderer, frost = DotRenderer(field), Frost(width, height, max(self._frost_feather, 1.5 * self.cell))  # soft edge must span a whole dot
        with self._lock:
            self._gen += 1
            self._field, self._renderer, self._frost = field, renderer, frost
            self._ready = None
        # The first frame at a new size is drawn right here, so a resize
        # never flashes an empty canvas. The render thread may still be on
        # an old-size frame; it uses the old objects and gets discarded.
        frame = _render_frame(_Job(self._gen, field, renderer, frost, self._last_regions, 0.0, True), self.voice)
        self._photo = ImageTk.PhotoImage("RGB", (width, height))
        if frame is not None:
            self._photo.paste(frame)
        self.itemconfigure(self._item, image=self._photo)
        self._regions = None
        self._needs_frame = False

    def _pointer(self, x, y=None) -> None:
        if self._field is not None:
            self._field.set_pointer(x, y)
            self._on_activity()

    def _on_click(self, event) -> None:
        if self._field is not None:
            self._field.add_ripple(event.x, event.y)
            self._on_activity()
