"""Blur-behind for UI drawn over the dot grid.

Two kinds of region, both composited into the grid image each frame:

- Halos: soft-edged areas (around message text, the status line, the
  composer) where the sharp dots are replaced by a dim, smooth glow of
  their own colour, so text stays readable while the grid keeps moving
  behind it.
- Panels: anti-aliased rounded rectangles (message bubbles, the composer)
  filled with a tint over that glow, with an optional 1px border.

The "blur" is cheap because the dots sit on a regular grid: averaging a
cell is just dot colour x the fraction of the cell its sprite covers
(DotRenderer.wash), and a bilinear upsample of that per-cell average is
a smooth glow — no per-pixel Gaussian. Halo masks are built at 1/4
resolution (they're feathered anyway) and only rebuilt when the layout
or scroll position changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

GLOW_GAIN = 0.55  # brightness of the glow relative to the cell average
_MASK_STEP = 4


@dataclass
class Panel:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1
    radius: float
    fill: tuple[int, int, int]
    alpha: float = 1.0  # < 1 lets the glow show through (frosted)
    border: tuple[int, int, int] | None = None
    border_width: float = 1.0


def _rrect_sdf(xs: np.ndarray, ys: np.ndarray, box, radius: float) -> np.ndarray:
    """Signed distance to a rounded rectangle (negative inside)."""
    x0, y0, x1, y1 = box
    hx, hy = (x1 - x0) / 2, (y1 - y0) / 2
    r = max(0.0, min(radius, hx, hy))
    qx = np.abs(xs - (x0 + x1) / 2) - (hx - r)
    qy = np.abs(ys - (y0 + y1) / 2) - (hy - r)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside - r


class Frost:
    def __init__(self, width: int, height: int, feather: float):
        self.width, self.height = width, height
        self.feather = feather
        self._mask: Image.Image | None = None  # "L", covers self._bbox
        self._bbox = (0, 0, 0, 0)  # x0, y0, x1, y1
        self._panels: list[tuple] = []

    @property
    def empty(self) -> bool:
        return self._mask is None and not self._panels

    def set_regions(self, halos: list[tuple], panels: list[Panel]) -> None:
        """halos: [(x0, y0, x1, y1, radius), ...] in canvas pixels."""
        w, h, f = self.width, self.height, self.feather
        self._mask = None
        halos = [b for b in halos if b[2] > b[0] and b[3] > b[1]]
        if halos:
            x0 = max(0, int(min(b[0] for b in halos) - f))
            y0 = max(0, int(min(b[1] for b in halos) - f))
            x1 = min(w, int(max(b[2] for b in halos) + f) + 1)
            y1 = min(h, int(max(b[3] for b in halos) + f) + 1)
            if x1 > x0 and y1 > y0:
                s = _MASK_STEP
                xs = (x0 + (np.arange((x1 - x0 + s - 1) // s) + 0.5) * s)[None, :].astype(np.float32)
                ys = (y0 + (np.arange((y1 - y0 + s - 1) // s) + 0.5) * s)[:, None].astype(np.float32)
                m = np.zeros((ys.shape[0], xs.shape[1]), np.float32)
                for x_0, y_0, x_1, y_1, r in halos:
                    d = _rrect_sdf(xs, ys, (x_0, y_0, x_1, y_1), r)
                    np.maximum(m, np.clip(1 - np.maximum(d, 0) / f, 0, 1), out=m)
                m = m * m * (3 - 2 * m)  # smoothstep
                small = Image.fromarray(np.rint(m * 255).astype(np.uint8), "L")
                self._mask = small.resize((x1 - x0, y1 - y0), Image.BILINEAR)
                self._bbox = (x0, y0, x1, y1)

        self._panels = []
        for p in panels:
            bx0, by0, bx1, by1 = p.box
            sx0, sy0 = max(0, int(bx0) - 1), max(0, int(by0) - 1)
            sx1, sy1 = min(w, int(bx1) + 2), min(h, int(by1) + 2)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            xs = (np.arange(sx0, sx1) + 0.5)[None, :].astype(np.float32)
            ys = (np.arange(sy0, sy1) + 0.5)[:, None].astype(np.float32)
            cov = np.clip(0.5 - _rrect_sdf(xs, ys, p.box, p.radius), 0, 1)
            fill_mask = Image.fromarray(np.rint(cov * p.alpha * 255).astype(np.uint8), "L")
            ring_mask = None
            if p.border is not None:
                bw = p.border_width
                inner = (bx0 + bw, by0 + bw, bx1 - bw, by1 - bw)
                cov_in = np.clip(0.5 - _rrect_sdf(xs, ys, inner, p.radius - bw), 0, 1)
                ring_mask = Image.fromarray(np.rint((cov - cov_in) * 255).astype(np.uint8), "L")
            self._panels.append(((sx0, sy0, sx1, sy1), tuple(p.fill), fill_mask, p.border, ring_mask))

    def compose(self, sharp: np.ndarray, wash_small: np.ndarray, field) -> Image.Image:
        """Sharp dot buffer + glow + panels -> a new image (sharp is untouched)."""
        out = Image.fromarray(sharp, "RGB").copy()
        if self._mask is not None:
            x0, y0, x1, y1 = self._bbox
            # pad one cell all round so the margins outside the dot grid
            # still map inside the source image
            small = np.pad(np.clip(wash_small * GLOW_GAIN, 0, 255).astype(np.uint8), ((1, 1), (1, 1), (0, 0)), mode="edge")
            glow = Image.fromarray(small, "RGB").filter(ImageFilter.GaussianBlur(0.7))
            c = field.cell
            gw, gh = small.shape[1], small.shape[0]
            src = (
                min(gw, max(0.0, (x0 - field.ox) / c + 1)),
                min(gh, max(0.0, (y0 - field.oy) / c + 1)),
                min(gw, max(0.0, (x1 - field.ox) / c + 1)),
                min(gh, max(0.0, (y1 - field.oy) / c + 1)),
            )
            glow = glow.resize((x1 - x0, y1 - y0), Image.BILINEAR, box=src)
            out.paste(glow, (x0, y0), self._mask)
        for box, fill, fill_mask, border, ring_mask in self._panels:
            out.paste(fill, box, fill_mask)
            if ring_mask is not None:
                out.paste(tuple(border), box, ring_mask)
        return out
