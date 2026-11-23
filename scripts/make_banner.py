"""Render docs/banner.png, the README banner, with the app's own dot grid
(ui/dot_grid.py) and blur (ui/frost.py), so it always matches the window.
Re-run after changing the grid's palette or look:

    venv\\Scripts\\python.exe scripts/make_banner.py

Windows only: it uses Segoe UI Variable from C:\\Windows\\Fonts, the same
face the app uses.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `ui` imports when run as a script

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from ui.dot_grid import (  # noqa: E402
    GRID_PEAK_BRIGHTNESS,
    GRID_SATURATION,
    SPEAKING,
    DotField,
    DotRenderer,
    VoiceLevels,
)
from ui.frost import Frost  # noqa: E402
from ui.theme import INK, SLATE  # noqa: E402

W, H = 1600, 480  # 2x of how GitHub shows it, for sharp high-DPI screens
CELL = 24
VOICE_CENTER_X = 0.68  # where the grid's warm centre sits, as a fraction of W
FONT_FILE = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "SegUIVar.ttf"
OUT = ROOT / "docs" / "banner.png"


def font(style: str, size: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT_FILE), size)
    f.set_variation_by_name(style)
    return f


def rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


def main() -> None:
    # The field is wider than the banner and cropped, so its centre (where
    # the voice glows warmest) lands right of the wordmark instead of under it.
    field_w = round(2 * VOICE_CENTER_X * W)
    field = DotField(field_w, H, CELL, saturation=GRID_SATURATION, peak_brightness=GRID_PEAK_BRIGHTNESS)
    field.set_state(SPEAKING)
    renderer = DotRenderer(field)

    # A frozen moment of speech: strong low bands, tapering highs.
    voice = VoiceLevels()
    bands = np.array([0.95, 0.9, 0.78, 0.62, 0.5, 0.4, 0.34, 0.3], np.float32)
    t = 0.0
    for _ in range(40):
        t += 1 / 30
        voice.push([(t, 0.85, bands)])
        voice.update(t, 1 / 30)
        idx, col = field.step(1 / 30, voice)
    renderer.render(idx, col)

    title = font("Semilight Display", 150)
    tagline = font("Regular", 34)
    x, title_top = 112, 132
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    tb = draw_probe.textbbox((x, title_top), "MIDAS", font=title)
    tag_y = tb[3] + 30
    gb = draw_probe.textbbox((x + 6, tag_y), "A voice assistant that runs on your PC.", font=tagline)

    # Same treatment as text in the app: the grid blurs into a dim glow behind it.
    frost = Frost(field_w, H, feather=1.5 * CELL)
    pad = 44
    frost.set_regions(
        [(x - pad, tb[1] - pad, max(tb[2], gb[2]) + pad, gb[3] + pad, 56)],
        [],
    )
    img = frost.compose(renderer.buffer, renderer.wash(idx, col), field).crop((0, 0, W, H))

    d = ImageDraw.Draw(img)
    d.text((x, title_top), "MIDAS", font=title, fill=rgb(INK))
    d.text((x + 6, tag_y), "A voice assistant that runs on your PC.", font=tagline, fill=rgb(SLATE))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, optimize=True)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
