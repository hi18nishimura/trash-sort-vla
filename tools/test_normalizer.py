"""Offline accuracy of the instruction normalizer (no simulator).

  source ~/lerobot-env/bin/activate && unset PYTHONPATH
  python tools/test_normalizer.py
"""
import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phrasing                                   # noqa: E402
from normalizer import InstructionNormalizer      # noqa: E402


def canonical_cases():
    for item in phrasing.CANONICAL_ITEM:
        for b in phrasing.COLORS:
            yield phrasing.canonical(item, b), item, b, {}


def unseen_cases():
    for item, words in phrasing.UNSEEN_ITEM.items():
        for w in words:
            for b, color in phrasing.COLORS.items():
                for bn in phrasing.UNSEEN_BIN:
                    for t in phrasing.UNSEEN_TEMPLATES:
                        meta = {"template": t.split("{")[0].strip() or t[:20],
                                "item word": w, "bin word": bn}
                        yield t.format(item=w, color=color, bin=bn), item, b, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()
    print(f"loading {args.model} ...", flush=True)
    norm = InstructionNormalizer(args.model)

    for name, cases in [("canonical", list(canonical_cases())), ("unseen", list(unseen_cases()))]:
        ok_total, times, fails = 0, [], []
        by_meta = defaultdict(lambda: [0, 0])
        for text, item, b, meta in cases:
            t0 = time.time()
            canon, p_item, p_bin, raw = norm.normalize(text)
            times.append(time.time() - t0)
            ok = (p_item == item and p_bin == b)
            ok_total += ok
            for k, v in meta.items():
                by_meta[(k, v)][0] += ok
                by_meta[(k, v)][1] += 1
            if not ok:
                fails.append((text, item, b, p_item, p_bin, raw))
        print(f"\n===== {name}: {ok_total}/{len(cases)} correct "
              f"({100 * ok_total / len(cases):.0f}%), mean {1000 * sum(times) / len(times):.0f} ms/call =====")
        for (k, v), (o, n) in sorted(by_meta.items()):
            print(f"  {k:10s} {v:32s} {o:3d}/{n:3d}")
        for text, item, b, p_item, p_bin, raw in fails[:30]:
            print(f"  FAIL  \"{text}\"  expected=({item}, {b})  got=({p_item}, {p_bin})  raw={raw!r}")


if __name__ == "__main__":
    main()
