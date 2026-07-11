#!/usr/bin/env python3
"""Generate Glyph-Matrix sprites for new Pokemon.

Faithful port of the author's tools/sprite_editor.html pipeline (so new sprites
match the existing Gen 1 ones):
  - trim transparency;
  - bake a WHITE background with a small border, then downscale to 25x25
    (single resize; LANCZOS keeps it crisp rather than blurry);
  - grayscale (0.299/0.587/0.114) + the tool's contrast curve (strength 175):
    luminance below 0.375 -> black, otherwise value*pow(lum, strength);
  - circular mask: pixels with distance from centre (12,12) <= 12.5 are kept
    (489 of 625), the rest transparent. No artificial outline.

matrix_XXXX.png reproduces the tool's LED preview: each lit pixel is a dot with
a 10% gap, drawn supersampled for anti-aliased grid lines.

Usage:  python3 scripts_gen2_sprites.py 152 153 154 ...
"""
import sys
import urllib.request
from io import BytesIO

import numpy as np
from PIL import Image

DRAWABLE = "app/src/main/res/drawable"
SPRITE_URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{id}.png"

SIZE = 25
CENTER = 12
RADIUS = 12.5
_yy, _xx = np.mgrid[0:SIZE, 0:SIZE]
MASK = np.sqrt((_xx - CENTER) ** 2 + (_yy - CENTER) ** 2) <= RADIUS   # 489 True

# Per-Pokemon framing overrides (fill = fraction of the 25px frame the creature
# spans). Wide/tall sprites can be shrunk a touch so they don't spill the circle.
PER_ID = {
    155: dict(fill=0.80),   # Cyndaquil: wide back-flame
}


def fetch(pokemon_id):
    with urllib.request.urlopen(SPRITE_URL.format(id=pokemon_id)) as r:
        return Image.open(BytesIO(r.read())).convert("RGBA")


def trim(im):
    a = np.array(im.split()[-1]); ys, xs = np.where(a > 0)
    if len(xs) == 0:
        return im
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def apply_contrast(gray, contrast=175):
    """Port of the tool's applyContrast(); pixels already at 255 stay white."""
    if contrast <= 0:
        return gray
    strength = contrast / 100.0
    lum = gray / 255.0
    if strength <= 1:
        factor = 1 - (1 - lum) * strength
    else:
        threshold = (strength - 1) * 0.5
        factor = np.where(lum < threshold, 0.0, np.power(lum, strength))
    value = np.floor(gray * factor)
    return np.where(gray < 255, value, gray)


def make_sprite(im, fill=0.86, contrast=175):
    im = trim(im)
    w, h = im.size
    side = max(1, round(max(w, h) / fill))          # white-padded square
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(im, ((side - w) // 2, (side - h) // 2), im)
    small = padded.resize((SIZE, SIZE), Image.LANCZOS)   # single, crisp downscale
    arr = np.array(small).astype(float)
    gray = arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114
    gray = apply_contrast(gray, contrast)
    L = np.clip(gray, 0, 255).astype("uint8")
    A = (MASK * 255).astype("uint8")
    return Image.merge("RGBA", [Image.fromarray(L)] * 3 + [Image.fromarray(A)])


def make_matrix(sprite25, out=300, gap_ratio=0.1, ss=4):
    """LED preview with per-pixel gaps (the grid), supersampled for smooth edges."""
    L = np.array(sprite25)[:, :, 0]
    A = np.array(sprite25)[:, :, 3]
    big = out * ss
    pixel = big / SIZE
    dot = pixel * (1 - gap_ratio)
    off = (pixel - dot) / 2
    canvas = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    px = canvas.load()
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas)
    for y in range(SIZE):
        for x in range(SIZE):
            if not MASK[y, x] or A[y, x] == 0:
                continue
            v = int(L[y, x])
            x0 = x * pixel + off; y0 = y * pixel + off
            d.rectangle([x0, y0, x0 + dot, y0 + dot], fill=(v, v, v, int(A[y, x])))
    return canvas.resize((out, out), Image.LANCZOS)


def main(ids):
    for i in ids:
        sp = make_sprite(fetch(i), **PER_ID.get(i, {}))
        assert (np.array(sp)[:, :, 3] > 0).sum() == 489
        sp.save(f"{DRAWABLE}/sprite_{i:04d}.png")
        make_matrix(sp).save(f"{DRAWABLE}/matrix_{i:04d}.png")
        print(f"generated {i}")


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]]
    if not ids:
        print("usage: python3 scripts_gen2_sprites.py <id> [<id> ...]"); sys.exit(1)
    main(ids)
