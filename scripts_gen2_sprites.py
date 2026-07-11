#!/usr/bin/env python3
"""Generate Glyph-Matrix sprites for new Pokemon in the author's style.

Author style (reverse-engineered from the existing Gen-1 sprites):
  - alpha is a fixed circular mask (489/625 = 78.2% of a 25x25 grid),
    taken from an existing sprite's alpha channel;
  - inside the circle the background is LIT (white); the creature is
    "etched" in darker tones keeping its natural dark outline;
  - high bimodal contrast (lots of near-white and near-black, few midtones).

Produces both sprite_XXXX.png (25x25, used for the real LED render + UI)
and matrix_XXXX.png (300x300 nearest-neighbour upscale, in-app preview only).

Usage:  python3 scripts_gen2_sprites.py 152 153 154 ...
"""
import sys
import urllib.request
from io import BytesIO

import numpy as np
from PIL import Image

DRAWABLE = "app/src/main/res/drawable"
MASK_SRC = f"{DRAWABLE}/sprite_0001.png"   # any existing author sprite -> circle mask
SPRITE_URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{id}.png"

MASK = np.array(Image.open(MASK_SRC).convert("RGBA"))[:, :, 3] > 10  # 25x25 bool, 489 True


def fetch(pokemon_id):
    url = SPRITE_URL.format(id=pokemon_id)
    with urllib.request.urlopen(url) as r:
        return Image.open(BytesIO(r.read())).convert("RGBA")


def trim(im):
    a = np.array(im.split()[-1]); ys, xs = np.where(a > 10)
    if len(xs) == 0:
        return im
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def _edge(body):
    """Body pixels adjacent to the (transparent) background -> silhouette outline."""
    nb = ~body
    nn = np.zeros_like(body)
    nn[1:, :] |= nb[:-1, :]; nn[:-1, :] |= nb[1:, :]
    nn[:, 1:] |= nb[:, :-1]; nn[:, :-1] |= nb[:, 1:]
    return body & nn


def make_sprite(im, size=25, fit=24, contrast=1.7, cap=225, detail_k=2.8):
    im = trim(im)
    w, h = im.size
    s = min(fit / w, fit / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2), im)
    r, g, b, al = [np.array(c).astype(float) for c in canvas.split()]
    lum = r * 0.299 + g * 0.587 + b * 0.114
    body = al > 60
    out = np.full((size, size), 255.0)          # lit (white) circle background
    if body.sum() > 0:
        lo, hi = np.percentile(lum[body], 2), np.percentile(lum[body], 98)
        n = np.clip((lum - lo) / (hi - lo), 0, 1) if hi > lo else np.clip(lum / 255, 0, 1)
        # adaptive gamma: pale creatures (high mean brightness) get their midtones
        # darkened to reveal internal detail; darker creatures are left untouched
        mean_b = float(n[body].mean())
        n = np.power(n, 1.0 + detail_k * max(0.0, mean_b - 0.5))
        n = np.clip((n - 0.5) * contrast + 0.5, 0, 1)   # S-curve -> bimodal
        # cap < 255 keeps even the brightest creature pixel below the lit background,
        # so pale Pokemon don't dissolve into the white circle
        out[body] = n[body] * cap
    out[_edge(body)] = 0                         # crisp dark silhouette outline
    out[~MASK] = 0
    outL = out.astype("uint8")
    outA = (MASK * 255).astype("uint8")
    return Image.merge("RGBA", [Image.fromarray(outL)] * 3 + [Image.fromarray(outA)])


def make_matrix(sprite25, scale=12):
    """300x300 in-app preview with the author's LED grid: 2px semi-transparent
    lines straddling every cell boundary, drawn only over lit pixels."""
    up = sprite25.resize((25 * scale, 25 * scale), Image.NEAREST)
    L = np.array(up)[:, :, 0]
    A = np.array(up)[:, :, 3].astype("int32")
    rr = np.arange(25 * scale) % scale
    grid = (rr == 0) | (rr == scale - 1)
    gmask = grid[:, None] | grid[None, :]
    A[(A > 0) & gmask] = 102
    return Image.merge("RGBA", [Image.fromarray(L)] * 3 + [Image.fromarray(A.astype("uint8"))])


# Per-Pokemon overrides for make_sprite kwargs (wide sprites that would touch the
# circle edge look off-centre at the default fit; shrink slightly to re-centre).
PER_ID = {
    155: dict(fit=22),   # Cyndaquil: wide back-flame -> smaller fit centres the body
}


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
