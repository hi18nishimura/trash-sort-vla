#!/usr/bin/env python3
"""Collect expert demonstrations for the trash-sorting VLA.

Each attempt: randomise all item poses (+-radius), send one sorting goal, record the
overhead camera + joint states at 10 Hz, judge success by the item's FINAL pose, and
save SUCCESSFUL & reasonably fast episodes as .npz. Resumes from what is already saved.

Run with the SYSTEM python (needs rclpy), after:
  ros2 launch pytamp_moveit_bridge trash_sort.launch.py gui:=false item:=none
Usage:
  python3 tools/collect_demos.py --target 2      # quick test (2 per pair)
  python3 tools/collect_demos.py --target 40     # real collection
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import rclpy
from sensor_msgs.msg import Image, JointState

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_goals import Runner, HOME, gz_detach, gz_set_pose  # noqa: E402

TRAIN_PAIRS = [("can", "bin_blue"), ("can", "bin_orange"),
               ("bottle", "bin_green"), ("bottle", "bin_orange"),
               ("paper", "bin_blue"), ("paper", "bin_green")]
HELDOUT_PAIRS = [("can", "bin_green"), ("bottle", "bin_blue"), ("paper", "bin_orange")]
ITEM_WORD = {"can": "can", "bottle": "plastic bottle", "paper": "paper cup"}
BIN_WORD = {"bin_blue": "blue", "bin_green": "green", "bin_orange": "orange"}
JOINTS = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1"]


def instruction(item, bin_id):
    return f"put the {ITEM_WORD[item]} in the {BIN_WORD[bin_id]} bin"


class Collector(Runner):
    def __init__(self):
        super().__init__()
        self.img = None
        self.q = None
        self.recording = False
        self.frames, self.states = [], []
        self.create_subscription(Image, "/overhead_camera/image", self._on_img, 5)
        self.create_subscription(JointState, "/joint_states", self._on_js, 50)
        self.create_timer(0.1, self._tick)   # 10 Hz recording

    def _on_img(self, msg):
        if msg.encoding not in ("rgb8", "bgr8"):
            return
        a = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.img = a[..., ::-1] if msg.encoding == "bgr8" else a

    def _on_js(self, msg):
        d = dict(zip(msg.name, msg.position))
        if all(j in d for j in JOINTS):
            self.q = np.array([d[j] for j in JOINTS], dtype=np.float32)

    def _tick(self):
        if self.recording and self.img is not None and self.q is not None:
            self.frames.append(self.img.copy())
            self.states.append(self.q.copy())

    def reset_random(self, rng, radius, layout_mode="home", home_prob=0.3):
        for name in HOME:
            gz_detach(name)
        self.spin_for(0.5)
        placed = {}
        spots = list(HOME.values())
        if layout_mode == "permute" and rng.random() >= home_prob:
            rng.shuffle(spots)          # items swap places: identity is no longer tied to position
        for name, (x, y, z) in zip(HOME, spots):
            p = (round(x + rng.uniform(-radius, radius), 4),
                 round(y + rng.uniform(-radius, radius), 4), z)
            if not gz_set_pose(name, p):
                self.spin_for(0.5)
                gz_set_pose(name, p)
            placed[name] = p
        self.spin_for(4.0)
        return placed


def count_saved(out_dir):
    c = Counter()
    for f in glob.glob(os.path.join(out_dir, "*.npz")):
        item, bin_id = os.path.basename(f).split("__")[:2]
        c[(item, bin_id)] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=40, help="successful episodes per train pair")
    ap.add_argument("--out", default=os.path.expanduser("~/projects/panda-tamp/data/raw"))
    ap.add_argument("--radius", type=float, default=0.015, help="item position jitter [m]")
    ap.add_argument("--max-sec", type=float, default=35.0, help="discard slower episodes")
    ap.add_argument("--max-attempts", type=int, default=2000)
    ap.add_argument("--max-consec-fail", type=int, default=3,
                    help="exit (code 2) after this many failures in a row")
    ap.add_argument("--ready-timeout", type=float, default=180.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--layout-mode", choices=["home", "permute"], default="home")
    ap.add_argument("--home-prob", type=float, default=0.3,
                    help="with --layout-mode permute: probability of keeping the usual layout")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    counts = count_saved(args.out)
    log_path = os.path.join(args.out, "attempts.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["time", "item", "bin", "kind", "sec", "frames", "saved", "detail"])

    rclpy.init()
    node = Collector()
    t0 = time.time()
    while node.pub.get_subscription_count() == 0 or not node.planner_has_world:
        if time.time() - t0 > args.ready_timeout:
            print("planner never became ready -> restart", flush=True)
            log_f.close()
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(2)
        rclpy.spin_once(node, timeout_sec=0.2)
    node.wait_ready()
    print("already saved: " + ", ".join(f"{i}->{b}:{counts[(i, b)]}" for i, b in TRAIN_PAIRS) + "\n")

    attempts = 0
    consec_fail = 0
    exit_code = 2   # 2 = please restart the simulator, 0 = all targets reached
    t_start = time.time()
    while attempts < args.max_attempts:
        pending = [p for p in TRAIN_PAIRS if counts[p] < args.target]
        if not pending:
            exit_code = 0
            break
        low = min(counts[p] for p in pending)
        item, bin_id = rng.choice([p for p in pending if counts[p] == low])  # keep pairs balanced
        attempts += 1

        placed = node.reset_random(rng, args.radius, args.layout_mode, args.home_prob)
        node.frames, node.states = [], []
        node.recording = True
        try:
            kind, xyz, dt, detail = node.run_goal(item, bin_id)
        except Exception as exc:  # e.g. Gazebo stopped answering
            node.recording = False
            print(f"simulator not responding ({type(exc).__name__}) -> restart", flush=True)
            break
        node.recording = False
        n = len(node.frames)

        saved = kind == "ok" and dt <= args.max_sec and n >= 20
        if saved:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(args.out, f"{item}__{bin_id}__{stamp}.npz")
            np.savez_compressed(
                path,
                images=np.stack(node.frames), states=np.stack(node.states),
                instruction=np.array(instruction(item, bin_id)),
                item=np.array(item), bin=np.array(bin_id),
                start_poses=np.array(json.dumps(placed)), sec=np.array(dt))
            counts[(item, bin_id)] += 1
        log.writerow([time.strftime("%H:%M:%S"), item, bin_id, kind, f"{dt:.0f}", n, int(saved),
                      " | ".join(detail).replace(",", " ")])
        log_f.flush()

        done = sum(min(counts[p], args.target) for p in TRAIN_PAIRS)
        goal = args.target * len(TRAIN_PAIRS)
        elapsed = (time.time() - t_start) / 60
        why = "" if saved else (f"  ({kind})" if kind != "ok" else f"  (too slow: {dt:.0f}s)")
        print(f"[try {attempts:4d}] {item:6s} -> {bin_id:10s}: {'SAVED' if saved else 'skip '}"
              f"{why}  frames={n}  progress {done}/{goal}  ({elapsed:.0f} min)", flush=True)
        if not saved:
            for d in detail:
                print(f"        {d}", flush=True)
        if kind == "stuck":
            print("item stuck in the gripper -> restart", flush=True)
            break
        consec_fail = 0 if saved else consec_fail + 1
        if consec_fail >= args.max_consec_fail:
            print(f"{consec_fail} failures in a row -> restart", flush=True)
            break

    print("\nfinished. saved per pair: " +
          ", ".join(f"{i}->{b}:{counts[(i, b)]}" for i, b in TRAIN_PAIRS))
    log_f.close()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
