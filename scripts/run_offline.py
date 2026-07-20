#!/usr/bin/env python3
"""Offline TAMP demo runner (V1–V3 — no ROS, no sim).

Thin wrapper around :func:`tamp.cli.offline_main` (kept for backward compatibility
with ``run.sh`` and the documented `python scripts/run_offline.py` invocation).
After `pip install -e .` you can also just run the `tamp-offline` console script.

    python scripts/run_offline.py --scene stack3
    python scripts/run_offline.py --scene replan --viz results/replan.png
"""
from tamp.cli import offline_main

if __name__ == "__main__":
    raise SystemExit(offline_main())
