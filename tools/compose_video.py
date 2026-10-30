"""Compose recorded episodes into stylish 1920x1080 (16:9) clips.

Layouts
  single : large oblique view on the left + the robot's camera (what the AI sees) as an inset
           on the right; the space below the inset is left free for captions.
  compare: two oblique views side by side (left = baseline, right = improved), each with an
           AI-view inset; free space above (instruction) and below (labels) for captions.
Text is added in the video editor; only a small "2倍速" badge is drawn (--no-badge to remove).

Run with the planner venv:
  .venv/bin/python tools/compose_video.py single  out.mp4 A.npz
  .venv/bin/python tools/compose_video.py compare out.mp4 A.npz B.npz
  .venv/bin/python tools/compose_video.py card    bg.png            # matching background for title cards
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
TOP, BOTTOM = (255, 255, 255), (246, 245, 239)   # white -> warm off-white
PRIMARY = (179, 176, 145)        # #B3B091  main colour
PRIMARY_DARK = (122, 119, 88)    # #7A7758  text / strong accents
NEUTRAL = (200, 203, 208)        # #C8CBD0  baseline (before improvement)
SLATE, BLUE, GREEN = NEUTRAL, PRIMARY, PRIMARY   # roles used by the layouts
FONTS = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def font(size):
    for p in FONTS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def background():
    g = np.linspace(0, 1, H)[:, None, None]
    arr = (np.array(TOP) * (1 - g) + np.array(BOTTOM) * g).repeat(W, axis=1).astype(np.uint8)
    img = Image.fromarray(arr)
    dots = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(dots)
    for y in range(20, H, 40):
        for x in range(20, W, 40):
            d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(179, 176, 145, 70))
    img.paste(dots, (0, 0), dots)
    ImageDraw.Draw(img).rectangle([0, H - 10, W, H], fill=PRIMARY)   # thin band at the bottom
    return img


def add_shadow(bg, box, radius=28, blur=26, offset=12, alpha=60):
    x, y, w, h = box
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([x, y + offset, x + w, y + h + offset], radius, fill=(0, 0, 0, alpha))
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    bg.paste(sh, (0, 0), sh)


def rounded_mask(w, h, r):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], r, fill=255)
    return m


def paste_frame(canvas, frame, box, radius, accent=None, border=None):
    x, y, w, h = box
    img = Image.fromarray(frame).resize((w, h), Image.LANCZOS)
    if accent:                                   # coloured strip on the top edge
        ImageDraw.Draw(img).rectangle([0, 0, w, 8], fill=accent)
    canvas.paste(img, (x, y), rounded_mask(w, h, radius))
    if border:
        ImageDraw.Draw(canvas).rounded_rectangle([x, y, x + w - 1, y + h - 1], radius, outline=border, width=3)


def badge(canvas, text, x_right=W - 48, y=36):
    if not text:
        return
    d = ImageDraw.Draw(canvas)
    f = font(28)
    bb = d.textbbox((0, 0), text, font=f)
    w, h = bb[2] - bb[0] + 36, bb[3] - bb[1] + 22
    d.rounded_rectangle([x_right - w, y, x_right, y + h], h // 2, fill=PRIMARY)
    d.text((x_right - w + 18, y + 8), text, font=f, fill=(255, 255, 255))


def load(path):
    d = np.load(path, allow_pickle=False)
    if "images" in d:                            # expert demo: overhead only
        return {"main": d["images"], "inset": None}
    show = d["showcase"] if d["showcase"].size else d["overhead"]
    return {"main": show, "inset": d["overhead"] if d["overhead"].size else None}


def layout_single(clip):
    square = clip["main"].shape[1] == clip["main"].shape[2]
    main = (64, 60, 960, 960) if square else (64, 60, 1280, 960)
    inset = None if clip["inset"] is None else (main[0] + main[2] + 64, 60, 448, 448)
    return [(clip, main, inset, BLUE)]


def layout_compare(a, b):
    w, h = 880, 660
    y = 270
    left, right = (48, y, w, h), (48 + w + 64, y, w, h)
    ins = 230
    li = (left[0] + w - ins - 20, y + h - ins - 20, ins, ins)
    ri = (right[0] + w - ins - 20, y + h - ins - 20, ins, ins)
    return [(a, left, li if a["inset"] is not None else None, SLATE),
            (b, right, ri if b["inset"] is not None else None, GREEN)]


def render(slots, out, speed, src_fps, hold_s, badge_text):
    bg = background()
    for _, box, inset, _ in slots:               # bake static shadows into the background
        add_shadow(bg, box)
        if inset:
            add_shadow(bg, inset, radius=18, blur=12, offset=6, alpha=70)
    fps = src_fps * speed
    n = max(len(c["main"]) for c, *_ in slots)
    hold = int(hold_s * fps)
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=16)
    for t in range(n + hold):
        canvas = bg.copy()
        for clip, box, inset, accent in slots:
            i = min(t, len(clip["main"]) - 1)
            paste_frame(canvas, clip["main"][i], box, 28, accent=accent)
            if inset:
                j = min(t, len(clip["inset"]) - 1)
                paste_frame(canvas, clip["inset"][j], inset, 18, border=PRIMARY)
        badge(canvas, badge_text)
        writer.append_data(np.asarray(canvas.convert("RGB")))
    writer.close()
    print(f"saved {out}  ({(n + hold) / fps:.1f}s at {speed:g}x, {W}x{H})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "compare", "card"])
    ap.add_argument("out")
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--src-fps", type=float, default=10.0)
    ap.add_argument("--hold", type=float, default=1.5)
    ap.add_argument("--no-badge", action="store_true")
    args = ap.parse_args()
    if args.mode == "card":
        background().save(args.out)
        print(f"saved {args.out}")
        return
    clips = [load(p) for p in args.clips]
    slots = layout_single(clips[0]) if args.mode == "single" else layout_compare(clips[0], clips[1])
    render(slots, args.out, args.speed, args.src_fps, args.hold,
           "" if args.no_badge else f"{args.speed:g}倍速")


if __name__ == "__main__":
    main()
