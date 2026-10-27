#!/usr/bin/env python3
"""Evaluate the VLA in Gazebo (needs: simulator item:=none, policy_server.py, vla_node.py).
Run with the SYSTEM python:
  python3 tools/eval_vla.py --trials 5                     # baseline condition
  python3 tools/eval_vla.py --trials 3 --layout swap       # items moved to each other's spots
  python3 tools/eval_vla.py --trials 3 --radius 0.03       # larger position noise
  python3 tools/eval_vla.py --trials 3 --phrasing unseen   # paraphrases never seen in training
Metrics:
  OK      : item released and resting in the instructed bin
  carried : item ended over the instructed bin, released or not (= the instruction was understood)
"""
import argparse
import os
import random
import sys
import time
from collections import defaultdict

import rclpy
from rclpy.node import Node
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
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--radius", type=float, default=0.015)
    ap.add_argument("--layout", choices=list(LAYOUTS), default="home")
    ap.add_argument("--phrasing", choices=["canonical", "unseen"], default="canonical")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pairs = {"all": TRAIN_PAIRS + HELDOUT_PAIRS, "train": TRAIN_PAIRS, "heldout": HELDOUT_PAIRS}[args.pairs]
    episodes = pairs * args.trials
    rng = random.Random(args.seed)
    rng.shuffle(episodes)
    text_rng = random.Random(args.seed + 1)

    rclpy.init()
    node = Evaluator()
    node.spin_for(2.0)
    ok_res, carried_res = defaultdict(list), defaultdict(list)
    for n, (item, bin_id) in enumerate(episodes, 1):
        text = phrasing.sample(item, bin_id, text_rng, args.phrasing)
        node.reset(rng, args.radius, LAYOUTS[args.layout])
        ok, carried, xyz, dt, events = node.episode(item, bin_id, text, args.timeout)
        ok_res[(item, bin_id)].append(ok)
        carried_res[(item, bin_id)].append(carried)
        tag = "train  " if (item, bin_id) in TRAIN_PAIRS else "HELDOUT"
        verdict = "OK " if ok else ("C  " if carried else "NG ")
        pos = "None" if xyz is None else f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})"
        grasped = sorted({e.split(":")[1] for e in events if e.startswith("grasp:")})
        released = any(e.startswith("release:") for e in events)
        print(f"[{n:2d}/{len(episodes)}] {tag} {item:6s} -> {bin_id:10s}: {verdict} final={pos} ({dt:.0f}s) "
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
