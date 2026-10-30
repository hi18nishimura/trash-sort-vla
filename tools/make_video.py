"""Render recorded episodes into an mp4 (optionally with Japanese captions).

Run with the planner venv (has Pillow + imageio):
  .venv/bin/python tools/make_video.py out.mp4 --clip A.npz                       # one clip
  .venv/bin/python tools/make_video.py out.mp4 --clip A.npz --clip B.npz          # side by side
Options:
  --view showcase|overhead   which camera (expert demos in data/raw have overhead only)
  --speed 2                  playback speed (also shown as a badge unless --no-badge)
  --no-caption / --no-badge  remove the top bar text; with both, there is no top bar at all
  --show-labels / --show-result   burn model names / OK-NG into the video (off by default)
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
RESULT_JA = {"OK": ("成功", (40, 170, 70)), "C": ("運んだが放さず", (230, 140, 20)),
             "NG": ("失敗", (210, 50, 50))}
BAR = 56       # height of the top bar
GAP = 4        # white line between side-by-side panels


def font(size):
    for p in FONTS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load(path, view):
    d = np.load(path, allow_pickle=False)
    if "images" in d:                              # expert demo (collect_demos.py)
        return d["images"], str(d["instruction"]), ""
    frames = d[view] if (view in d and d[view].size) else d["overhead"]
    return frames, str(d["instruction"]), str(d["result"])


def panel(frame, width, label="", result=""):
    img = Image.fromarray(frame)
    img = img.resize((width, int(round(frame.shape[0] * width / frame.shape[1] / 2)) * 2))
    draw = ImageDraw.Draw(img)
    f = font(max(16, width // 26))
    if label:
        box = draw.textbbox((0, 0), label, font=f)
        draw.rectangle([0, img.height - (box[3] - box[1]) - 18, box[2] - box[0] + 20, img.height], fill=(0, 0, 0))
        draw.text((10, img.height - (box[3] - box[1]) - 12), label, font=f, fill=(255, 255, 255))
    if result in RESULT_JA:
        text, color = RESULT_JA[result]
        box = draw.textbbox((0, 0), text, font=f)
        x0 = img.width - (box[2] - box[0]) - 24
        draw.rectangle([x0, 8, img.width - 8, 8 + (box[3] - box[1]) + 16], fill=color)
        draw.text((x0 + 8, 12), text, font=f, fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--clip", action="append", required=True)
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--view", choices=["showcase", "overhead"], default="showcase")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--src-fps", type=float, default=10.0)
    ap.add_argument("--caption", default=None, help="top bar text (default: the instruction)")
    ap.add_argument("--no-caption", action="store_true")
    ap.add_argument("--no-badge", action="store_true")
    ap.add_argument("--show-labels", action="store_true")
    ap.add_argument("--show-result", action="store_true")
    ap.add_argument("--width", type=int, default=None, help="width of each panel (default 960 / n_clips)")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds to hold the last frame")
    args = ap.parse_args()

    clips = [load(p, args.view) for p in args.clip]
    labels = (args.label + [""] * len(clips))[: len(clips)] if args.show_labels else [""] * len(clips)
    width = args.width or (960 // len(clips)) // 16 * 16
    caption = "" if args.no_caption else (args.caption if args.caption is not None
                                          else f"指示:「{clips[0][1]}」")
    badge = "" if args.no_badge else f"{args.speed:g}倍速"
    bar = BAR if (caption or badge) else 0
    out_fps = args.src_fps * args.speed
    n = max(len(c[0]) for c in clips)
    hold = int(args.hold * out_fps)
    cap_font, badge_font = font(26), font(22)

    writer = imageio.get_writer(args.out, fps=out_fps, codec="libx264", quality=8, macro_block_size=16)
    for t in range(n + hold):
        final = t >= n - 1
        panels = [panel(c[0][min(t, len(c[0]) - 1)], width, lab,
                        c[2] if (args.show_result and final) else "")
                  for c, lab in zip(clips, labels)]
        h = max(p.height for p in panels)
        total_w = width * len(panels) + GAP * (len(panels) - 1)
        canvas = Image.new("RGB", (total_w, h + bar), (255, 255, 255) if len(panels) > 1 else (20, 20, 20))
        if bar:
            ImageDraw.Draw(canvas).rectangle([0, 0, total_w, bar], fill=(20, 20, 20))
        for i, p in enumerate(panels):
            canvas.paste(p, (i * (width + GAP), bar))
        draw = ImageDraw.Draw(canvas)
        if caption:
            draw.text((14, 12), caption, font=cap_font, fill=(255, 255, 255))
        if badge:
            bb = draw.textbbox((0, 0), badge, font=badge_font)
            draw.text((canvas.width - (bb[2] - bb[0]) - 14, 15), badge, font=badge_font, fill=(255, 210, 80))
        arr = np.asarray(canvas)
        arr = arr[: arr.shape[0] // 16 * 16, : arr.shape[1] // 16 * 16]
        writer.append_data(arr)
    writer.close()
    print(f"saved {args.out}  ({n} frames -> {(n + hold) / out_fps:.1f}s at {args.speed:g}x)")


if __name__ == "__main__":
    main()
