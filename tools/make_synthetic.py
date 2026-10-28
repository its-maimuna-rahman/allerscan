"""Builds synthetic 'photos' of ingredient labels (for testing the pipeline without real photos)."""
import json, random
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LABELS = [
 ("INGREDIENTS: Sugar, wheat flour, skimmed milk powder, hazelnuts (13%), cocoa butter, soy lecithin, "
  "vanillin. May contain traces of peanuts and sesame seeds.",
  {"wheat", "cow_milk", "hazelnut", "soybean", "peanut", "sesame_seed"}),
 ("Ingredients: Water, tomato paste, celery, onion, mustard seed, salt, spice extract. "
  "Produced in a facility that also handles fish and egg.",
  {"celery", "mustard_seed"}),
 ("INGREDIENTS: Rice flour, palm oil, whey powder, almonds, cashews, egg white, salt. Contains: Milk, Egg, Tree Nuts.",
  {"cow_milk", "almond", "cashew", "chicken_egg"}),
]

def render(text, width=900, font_px=26, bg=255, fg=0):
    font = ImageFont.truetype(FONT, font_px)
    img = Image.new("L", (width, 10), bg); d = ImageDraw.Draw(img)
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) > width - 80: lines.append(cur); cur = w
        else: cur = t
    lines.append(cur)
    h = 60 + len(lines) * int(font_px * 1.5)
    img = Image.new("L", (width, h), bg); d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines): d.text((40, 30 + i * int(font_px * 1.5)), ln, font=font, fill=fg)
    return np.array(img)

def to_photo(gray, rot=0, skew=0.0, blur=0.0, noise=0.0, margin=120, bg_dark=False, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    g = gray if not bg_dark else 255 - gray
    g = cv2.copyMakeBorder(g, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=int(g[0, 0]))
    if scale != 1.0: g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if skew:
        h, w = g.shape; m = cv2.getRotationMatrix2D((w/2, h/2), skew, 1.0)
        g = cv2.warpAffine(g, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rot: g = np.rot90(g, k=rot // 90 * -1)   # rot=90 -> clockwise
    if blur: g = cv2.GaussianBlur(g, (0, 0), blur)
    if noise: g = np.clip(g.astype(np.float32) + rng.normal(0, noise, g.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(np.ascontiguousarray(g), cv2.COLOR_GRAY2RGB)

def build(out_dir=None):
    out = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "scan_data"
    out.mkdir(exist_ok=True)
    cases = []
    cfgs = [dict(), dict(skew=4.0), dict(rot=90), dict(rot=180), dict(rot=270), dict(bg_dark=True),
            dict(blur=1.2, noise=8), dict(scale=0.55, skew=-3.0), dict(skew=-6.0, rot=90, noise=6)]
    k = 0
    for li, (text, expected) in enumerate(LABELS):
        for ci, cfg in enumerate(cfgs):
            name = f"syn_{li}_{ci}"
            img = to_photo(render(text, font_px=26), seed=k, **cfg); k += 1
            cv2.imwrite(str(out / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            (out / f"{name}.txt").write_text(text, encoding="utf-8")
            cases.append(name)
    return cases
if __name__ == "__main__":
    print(len(build()), "synthetic photos written to scan_data/")
