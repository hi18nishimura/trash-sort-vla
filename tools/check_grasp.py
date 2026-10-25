"""Offline feasibility checker for the trash-sorting scene (no ROS, no Gazebo).

  combos  : can each of the 9 (item, bin) pairs be planned in the current layout?
  isolate : why does one pair fail? (other objects removed / other bins removed)
  scan    : grid-scan one item's position -> map of where it can be picked & placed
"""
import argparse
import copy
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tamp.tamp_solver import TampSolver          # noqa: E402
from tamp.types import Pose                      # noqa: E402
from benchmarks import scene_trash               # noqa: E402

ITEMS = {"can": "can_1", "bottle": "bottle_1", "paper": "paper_1"}
BINS = ["bin_blue", "bin_green", "bin_orange"]


def _pose_attr(obj):
    """Name of the attribute that holds the object's Pose."""
    for name, val in vars(obj).items():
        if isinstance(val, Pose):
            return name
    raise RuntimeError(f"no Pose attribute on {obj}")


def make_variant(obj_id, xyz=None, keep_others=True, keep_bins=None):
    """Copy of the trash world with one item moved / others removed."""
    world = scene_trash.make_world()
    objs = []
    for o in world.objects:
        if o.object_id == obj_id:
            if xyz is not None:
                o = copy.copy(o)
                object.__setattr__(o, _pose_attr(o), Pose(position=tuple(xyz)))
        elif not keep_others:
            continue
        objs.append(o)
    surfs = [s for s in world.surfaces
             if keep_bins is None or s.surface_id == "workspace" or s.surface_id in keep_bins]
    w = copy.copy(world)
    object.__setattr__(w, "objects", objs)
    object.__setattr__(w, "surfaces", surfs)
    return w


def solve(world, obj_id, bin_id):
    t = time.time()
    res = TampSolver().solve(world, {"type": "on", "object": obj_id, "surface": bin_id})
    return res, time.time() - t


def cmd_combos(_args):
    print("9 combos in the current layout (all items and bins present)\n")
    rows = []
    for item, oid in ITEMS.items():
        cells = []
        for b in BINS:
            res, dt = solve(make_variant(oid), oid, b)
            cells.append("OK" if res.success else "NG")
            print(f"  {item:7s} -> {b:10s}: {'OK' if res.success else 'NG'}  ({dt:.1f}s)"
                  + ("" if res.success else f"  reason={res.reason}"))
        rows.append((item, cells))
    print("\n          " + "  ".join(f"{b:10s}" for b in BINS))
    for item, cells in rows:
        print(f"  {item:7s} " + "  ".join(f"{c:10s}" for c in cells))


def cmd_isolate(args):
    oid = ITEMS[args.item]
    print(f"Isolating why {args.item} -> {args.bin} fails\n")
    tests = [
        ("full scene", dict()),
        ("other items removed", dict(keep_others=False)),
        ("other bins removed", dict(keep_bins=[args.bin])),
        ("only this item + this bin", dict(keep_others=False, keep_bins=[args.bin])),
    ]
    for label, kw in tests:
        res, dt = solve(make_variant(oid, **kw), oid, args.bin)
        print(f"  {label:28s}: {'OK' if res.success else 'NG'}  ({dt:.1f}s)"
              + ("" if res.success else f"  reason={res.reason}"))


def cmd_scan(args):
    oid = ITEMS[args.item]
    xs = [round(args.xmin + i * args.step, 3)
          for i in range(int(round((args.xmax - args.xmin) / args.step)) + 1)]
    ys = [round(args.ymin + i * args.step, 3)
          for i in range(int(round((args.ymax - args.ymin) / args.step)) + 1)]
    print(f"Scanning {args.item} -> {args.bin} (other items removed), z=0.445")
    print("  O = can pick & place,  . = cannot\n")
    print("   x \\ y " + " ".join(f"{y:+.2f}" for y in ys))
    for x in xs:
        line = []
        for y in ys:
            res, _ = solve(make_variant(oid, xyz=(x, y, 0.445), keep_others=False), oid, args.bin)
            line.append("  O  " if res.success else "  .  ")
        print(f"  {x:.3f} " + " ".join(line), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("combos")
    p = sub.add_parser("isolate")
    p.add_argument("--item", default="bottle", choices=ITEMS)
    p.add_argument("--bin", default="bin_green", choices=BINS)
    p = sub.add_parser("scan")
    p.add_argument("--item", default="bottle", choices=ITEMS)
    p.add_argument("--bin", default="bin_green", choices=BINS)
    p.add_argument("--xmin", type=float, default=0.40)
    p.add_argument("--xmax", type=float, default=0.50)
    p.add_argument("--ymin", type=float, default=-0.25)
    p.add_argument("--ymax", type=float, default=0.25)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
                        format="[%(name)s] %(message)s")
    {"combos": cmd_combos, "isolate": cmd_isolate, "scan": cmd_scan}[args.mode](args)


if __name__ == "__main__":
    main()
