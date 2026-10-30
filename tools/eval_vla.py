#!/usr/bin/env python3
"""Evaluate the VLA in Gazebo (needs: simulator item:=none, policy_server.py, vla_node.py).
Run with the SYSTEM python:
  python3 tools/eval_vla.py --trials 5                     # baseline condition
  python3 tools/eval_vla.py --trials 3 --layout swap       # items moved to each other's spots
  python3 tools/eval_vla.py --trials 3 --phrasing unseen   # paraphrases never seen in training
Recording for the video:
  python3 tools/eval_vla.py --trials 2 --only can:bin_blue --record outputs/rec/shot2 --tag v1
  python3 tools/eval_vla.py --trials 2 --only can:bin_blue --text "缶を青い箱に入れて" --record ... --tag v1_llm
Metrics:
  OK      : item released and resting in the instructed bin
  carried : item ended over the instructed bin, released or not (= the instruction was understood)
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_goals import HOME, BINS, BIN_HALF, gz_detach, gz_set_pose, gz_model_pose, in_bin  # noqa: E402
from collect_demos import TRAIN_PAIRS, HELDOUT_PAIRS                                      # noqa: E402
from vla_node import dur, ITEMS, ARM                                                      # noqa: E402
import phrasing                                                                           # noqa: E402

HOME_Q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]   # Gazebo start pose of the arm
LAYOUTS = {
    "home": dict(HOME),
    "swap": {"can": HOME["paper"], "bottle": HOME["can"], "paper": HOME["bottle"]},
}


def over_bin(xyz, bin_id):
    bx, by = BINS[bin_id]
    return xyz is not None and abs(xyz[0] - bx) <= BIN_HALF and abs(xyz[1] - by) <= BIN_HALF and xyz[2] > 0.40


def _to_np(msg):
    a = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    return a[..., ::-1] if msg.encoding == "bgr8" else a


class Recorder(Node):
    """Records both cameras at 10 Hz in a background thread (never blocked by the evaluator)."""

    def __init__(self):
        super().__init__("vla_recorder")
        self.latest = {"overhead": None, "showcase": None}
        self.frames = {"overhead": [], "showcase": []}
        self.active = False
        self.lock = threading.Lock()
        self.create_subscription(Image, "/overhead_camera/image", lambda m: self._on(m, "overhead"), 2)
        self.create_subscription(Image, "/showcase_camera/image", lambda m: self._on(m, "showcase"), 2)
        self.create_timer(0.1, self._tick)

    def _on(self, msg, key):
        if msg.encoding in ("rgb8", "bgr8"):
            self.latest[key] = _to_np(msg)

    def _tick(self):
        with self.lock:
            if self.active:
                for k, v in self.latest.items():
                    if v is not None:
                        self.frames[k].append(v.copy())

    def start(self):
        with self.lock:
            self.frames = {"overhead": [], "showcase": []}
            self.active = True

    def stop(self):
        with self.lock:
            self.active = False
            return self.frames


class Evaluator(Node):
    def __init__(self):
        super().__init__("vla_evaluator")
        self.task_pub = self.create_publisher(String, "/vla/task", 10)
        self.arm_pub = self.create_publisher(JointTrajectory, "/panda_arm_controller/joint_trajectory", 10)
        self.grasp_pub = self.create_publisher(String, "/grasp_cmd", 10)
        self.create_subscription(String, "/vla/status", self._on_status, 50)
        self.events = []

    def _on_status(self, msg):
        self.events.append(msg.data)

    def spin_for(self, sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def reset(self, rng, radius, positions):
        self.task_pub.publish(String(data="stop"))
        for oid in ITEMS.values():
            self.grasp_pub.publish(String(data=f"release:{oid}"))
        for name in HOME:
            gz_detach(name)
        jt = JointTrajectory()
        jt.joint_names = ARM
        p = JointTrajectoryPoint()
        p.positions = HOME_Q
        p.time_from_start = dur(3.0)
        jt.points = [p]
        self.arm_pub.publish(jt)
        self.spin_for(4.0)
        for name, (x, y, z) in positions.items():
            gz_set_pose(name, (round(x + rng.uniform(-radius, radius), 4),
                               round(y + rng.uniform(-radius, radius), 4), z))
        self.spin_for(3.0)

    def episode(self, item, bin_id, text, timeout):
        self.events = []
        self.task_pub.publish(String(data=text))
        t0 = time.time()
        success = False
        while time.time() - t0 < timeout:
            self.spin_for(4.0)
            released = any(e.startswith("release:") for e in self.events)
            xyz, _ = gz_model_pose(item)
            if released and xyz is not None and in_bin(xyz, bin_id):
                self.spin_for(2.0)                      # confirm it stays there
                xyz, _ = gz_model_pose(item)
                if xyz is not None and in_bin(xyz, bin_id):
                    success = True
                    break
        self.task_pub.publish(String(data="stop"))
        xyz, _ = gz_model_pose(item)
        return success, over_bin(xyz, bin_id), xyz, time.time() - t0, list(self.events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--pairs", choices=["all", "train", "heldout"], default="all")
    ap.add_argument("--only", action="append", default=None, help="item:bin, e.g. can:bin_blue (repeatable)")
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--radius", type=float, default=0.015)
    ap.add_argument("--layout", choices=list(LAYOUTS), default="home")
    ap.add_argument("--phrasing", choices=["canonical", "unseen"], default="canonical")
    ap.add_argument("--text", default=None, help="use this exact instruction (e.g. Japanese)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", default=None, help="folder to save camera frames of every episode")
    ap.add_argument("--tag", default="model", help="model name written into recordings")
    args = ap.parse_args()
    if args.only:
        pairs = [tuple(p.split(":")) for p in args.only]
    else:
        pairs = {"all": TRAIN_PAIRS + HELDOUT_PAIRS, "train": TRAIN_PAIRS, "heldout": HELDOUT_PAIRS}[args.pairs]
    episodes = pairs * args.trials
    rng = random.Random(args.seed)
    rng.shuffle(episodes)
    text_rng = random.Random(args.seed + 1)

    rclpy.init()
    node = Evaluator()
    rec = None
    if args.record:
        os.makedirs(args.record, exist_ok=True)
        rec = Recorder()
        ex = SingleThreadedExecutor()
        ex.add_node(rec)
        threading.Thread(target=ex.spin, daemon=True).start()
    node.spin_for(2.0)
    ok_res, carried_res = defaultdict(list), defaultdict(list)
    for n, (item, bin_id) in enumerate(episodes, 1):
        text = args.text or phrasing.sample(item, bin_id, text_rng, args.phrasing)
        node.reset(rng, args.radius, LAYOUTS[args.layout])
        if rec:
            rec.start()
        ok, carried, xyz, dt, events = node.episode(item, bin_id, text, args.timeout)
        verdict = "OK" if ok else ("C" if carried else "NG")
        if rec:
            node.spin_for(1.0)                          # a short tail after the episode
            fr = rec.stop()
            path = os.path.join(args.record, f"{args.tag}__{item}__{bin_id}__{args.layout}__{n:02d}__{verdict}.npz")
            np.savez_compressed(
                path,
                overhead=np.stack(fr["overhead"]) if fr["overhead"] else np.zeros((0,)),
                showcase=np.stack(fr["showcase"]) if fr["showcase"] else np.zeros((0,)),
                instruction=np.array(text), tag=np.array(args.tag), result=np.array(verdict),
                item=np.array(item), bin=np.array(bin_id), layout=np.array(args.layout),
                events=np.array(json.dumps(events)))
            print(f"        saved {path}  ({len(fr['overhead'])} overhead / {len(fr['showcase'])} showcase frames)",
                  flush=True)
        ok_res[(item, bin_id)].append(ok)
        carried_res[(item, bin_id)].append(carried)
        tag = "train  " if (item, bin_id) in TRAIN_PAIRS else "HELDOUT"
        pos = "None" if xyz is None else f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})"
        grasped = sorted({e.split(":")[1] for e in events if e.startswith("grasp:")})
        released = any(e.startswith("release:") for e in events)
        print(f"[{n:2d}/{len(episodes)}] {tag} {item:6s} -> {bin_id:10s}: {verdict:3s} final={pos} ({dt:.0f}s) "
              f"grasped={grasped} released={released}  \"{text}\"", flush=True)

    def rate(res, ps):
        r = [x for p in ps for x in res.get(p, [])]
        return f"{sum(r)}/{len(r)}" if r else "-"
    print(f"\nconditions: layout={args.layout} radius={args.radius} phrasing={args.phrasing}")
    print(f"OK      : train {rate(ok_res, TRAIN_PAIRS):6s}  heldout {rate(ok_res, HELDOUT_PAIRS)}")
    print(f"carried : train {rate(carried_res, TRAIN_PAIRS):6s}  heldout {rate(carried_res, HELDOUT_PAIRS)}"
          "   (item ended over the instructed bin, released or not)")
    print("\n  OK (carried)  " + "  ".join(f"{b:12s}" for b in BINS))
    for item in HOME:
        cells = []
        for b in BINS:
            r, c = ok_res.get((item, b), []), carried_res.get((item, b), [])
            mark = "*" if (item, b) in HELDOUT_PAIRS else " "
            cells.append(f"{sum(r)}/{len(r)} ({sum(c)}){mark}" if r else "-")
        print(f"  {item:12s} " + "  ".join(f"{c:12s}" for c in cells))
    print("  (* = held-out pair, never seen in training)")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
