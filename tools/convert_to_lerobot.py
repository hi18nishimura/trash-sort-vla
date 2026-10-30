"""Convert raw .npz demos (tools/collect_demos.py) into a LeRobot dataset.

Run in the LeRobot env:
  source ~/lerobot-env/bin/activate && unset PYTHONPATH
  python tools/convert_to_lerobot.py --stats-only     # just report what would be kept
  python tools/convert_to_lerobot.py                  # write the dataset

Per frame:
  observation.images.overhead : (320, 320, 3) uint8
  observation.state           : 7 arm joints + finger position (m)   (as recorded)
  action                      : NEXT frame's 7 arm joints (median-filtered) + gripper_open (1/0)
  task                        : the language instruction

Joint states contain short physics glitches (the DetachableJoint weld yanking the arm for
1-2 frames). Actions and the keep/drop decision use a temporal median filter so the policy
does not learn those glitches.
"""
import argparse
import random
import glob
import os
import shutil
from collections import Counter, defaultdict

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FPS = 10
STATE_NAMES = [f"joint{i}" for i in range(1, 8)] + ["finger"]
ACTION_NAMES = [f"joint{i}" for i in range(1, 8)] + ["gripper_open"]
OPEN_THRESHOLD = 0.039   # finger > this  -> gripper is (being) opened


def smooth(states, k):
    """Temporal median filter (window k frames, edges padded)."""
    if k <= 1:
        return states.copy()
    p = k // 2
    padded = np.pad(states, ((p, p), (0, 0)), mode="edge")
    return np.median(sliding_window_view(padded, k, axis=0), axis=-1)


def trim(arm, still=1e-3, pad=5):
    """Keep from `pad` frames before the arm starts moving to `pad` frames after it stops."""
    d = np.abs(np.diff(arm, axis=0)).max(axis=1)
    moving = np.where(d > still)[0]
    if len(moving) == 0:
        return 0, len(arm)
    return max(0, moving[0] - pad), min(len(arm), moving[-1] + 2 + pad)


def max_jump(arm):
    """(largest |dq| per frame over all arm joints, index of that joint)."""
    d = np.abs(np.diff(arm, axis=0))
    j = int(np.unravel_index(np.argmax(d), d.shape)[1])
    return float(d.max()), j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", default=[os.path.expanduser("~/projects/panda-tamp/data/raw")],
                    help="one or more raw folders; processed in the given order")
    ap.add_argument("--repeat", type=int, nargs="+", default=None,
                    help="how many times to use each --raw folder, e.g. 1 3")
    ap.add_argument("--root", default=os.path.expanduser("~/projects/panda-tamp/data/lerobot/trash_sort"))
    ap.add_argument("--repo-id", default="local/trash_sort")
    ap.add_argument("--smooth", type=int, default=5, help="median filter window [frames], odd")
    ap.add_argument("--max-jump", type=float, default=0.3, help="max |dq| per frame after smoothing [rad]")
    ap.add_argument("--min-len", type=int, default=30, help="min frames after trimming")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--no-video", action="store_true", help="store PNG frames instead of mp4")
    ap.add_argument("--phrasing", choices=["canonical", "train"], default="canonical",
                    help="train = replace instructions with paraphrases (v2)")
    ap.add_argument("--para-prob", type=float, default=0.7, help="probability of a paraphrase")
    ap.add_argument("--phr-seed", type=int, default=0)
    args = ap.parse_args()

    repeats = args.repeat or [1] * len(args.raw)
    assert len(repeats) == len(args.raw), "--repeat needs one number per --raw folder"
    files = []
    for raw_dir, rep_n in zip(args.raw, repeats):   # keep folder order (v2's episodes first)
        files += sorted(glob.glob(os.path.join(raw_dir, "*.npz"))) * rep_n
    rows = []
    raw_joint = Counter()
    for f in files:
        st = np.load(f)["states"].astype(np.float64)
        sm = smooth(st, args.smooth)
        a, b = trim(sm[:, :7])
        raw_j, raw_idx = max_jump(st[a:b, :7]) if b - a > 1 else (0.0, 0)
        sm_j, _ = max_jump(sm[a:b, :7]) if b - a > 1 else (0.0, 0)
        raw_joint[raw_idx + 1] += 1
        pair = "__".join(os.path.basename(f).split("__")[:2])
        rows.append((f, pair, a, b, raw_j, sm_j))

    raw_js = np.array([r[4] for r in rows])
    sm_js = np.array([r[5] for r in rows])
    print(f"{len(files)} raw episodes (median filter window = {args.smooth} frames)")
    print(f"  max |dq|/frame RAW     : median {np.median(raw_js):.3f}  90% {np.percentile(raw_js, 90):.3f} rad")
    print(f"  max |dq|/frame SMOOTHED: median {np.median(sm_js):.3f}  90% {np.percentile(sm_js, 90):.3f} rad")
    print("  joint with the largest RAW jump: " +
          ", ".join(f"joint{j}:{c}" for j, c in sorted(raw_joint.items())))
    print("\n  episodes kept at different thresholds (smoothed, length >= %d):" % args.min_len)
    for th in (0.1, 0.2, 0.3, 0.5):
        k = sum(1 for r in rows if r[5] <= th and r[3] - r[2] >= args.min_len)
        print(f"    max |dq| <= {th:.1f} rad : {k}/{len(rows)}")

    kept = [r for r in rows if r[5] <= args.max_jump and r[3] - r[2] >= args.min_len]
    report = defaultdict(lambda: [0, 0, []])
    for f, pair, a, b, _, sm_j in rows:
        report[pair][1] += 1
    for f, pair, a, b, _, _ in kept:
        report[pair][0] += 1
        report[pair][2].append(b - a)
    print(f"\n  per pair at threshold {args.max_jump}:")
    for pair, (k, n, lens) in sorted(report.items()):
        print(f"    {pair:22s} kept {k:2d}/{n:2d}   mean length {np.mean(lens) / FPS if lens else 0:.1f}s")
    print(f"\nTOTAL kept {len(kept)}/{len(files)}")
    if args.stats_only:
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if os.path.exists(args.root):
        shutil.rmtree(args.root)
    features = {
        "observation.images.overhead": {"dtype": "image" if args.no_video else "video",
                                        "shape": (320, 320, 3),
                                        "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (8,), "names": ACTION_NAMES},
    }
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=features, root=args.root,
                               robot_type="panda", use_videos=not args.no_video,
                               image_writer_threads=4)
    import phrasing
    phr_rng = random.Random(args.phr_seed)
    for i, (f, pair, a, b, _, _) in enumerate(kept, 1):
        d = np.load(f)
        imgs = d["images"][a:b]
        st = d["states"][a:b].astype(np.float32)
        sm = smooth(d["states"].astype(np.float64), args.smooth)[a:b].astype(np.float32)
        task = str(d["instruction"])
        if args.phrasing == "train" and phr_rng.random() < args.para_prob:
            task = phrasing.sample(str(d["item"]), str(d["bin"]), phr_rng, "train")
        for t in range(len(st)):
            nxt = sm[min(t + 1, len(sm) - 1)]
            nxt_finger = st[min(t + 1, len(st) - 1), 7]
            action = np.concatenate([nxt[:7], [1.0 if nxt_finger > OPEN_THRESHOLD else 0.0]]).astype(np.float32)
            ds.add_frame({"observation.images.overhead": imgs[t],
                          "observation.state": st[t],
                          "action": action,
                          "task": task})
        ds.save_episode()
        print(f"[{i}/{len(kept)}] {os.path.basename(f)}  frames={len(st)}  '{task}'", flush=True)
    if hasattr(ds, "finalize"):
        ds.finalize()
    print(f"\ndataset written to {args.root}")


if __name__ == "__main__":
    main()
