#!/usr/bin/env python3
"""Measure the expert's success rate per (item, bin), with a FULL reset before every episode
(the same condition as data collection). Each episode is judged by the item's FINAL pose in
Gazebo, and failures are classified with the planner/executor reasons.

Run with the SYSTEM python (needs rclpy), after:
  ros2 launch pytamp_moveit_bridge trash_sort.launch.py gui:=true item:=none
Usage:
  python3 tools/run_goals.py --trials 3                 # all items
  python3 tools/run_goals.py --trials 3 --items paper   # only paper
"""
import argparse
import json
import math
import random
import re
import subprocess
import time
from collections import Counter, defaultdict

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rcl_interfaces.msg import Log

WORLD = "tabletop"
HOME = {"can": (0.45, 0.20, 0.445), "bottle": (0.40, -0.10, 0.445), "paper": (0.48, -0.22, 0.445)}
BINS = {"bin_blue": (0.56, 0.13), "bin_green": (0.56, 0.0), "bin_orange": (0.56, -0.13)}
BIN_HALF = 0.05          # bins are 10 cm square
TIMEOUT = 150.0          # seconds per goal


def gz_set_pose(name, xyz):
    req = (f'name: "{name}", position: {{x: {xyz[0]}, y: {xyz[1]}, z: {xyz[2]}}}, '
           f'orientation: {{x: 0, y: 0, z: 0, w: 1}}')
    r = subprocess.run(["gz", "service", "-s", f"/world/{WORLD}/set_pose",
                        "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                        "--timeout", "3000", "--req", req], capture_output=True, text=True)
    return "true" in r.stdout


def gz_detach(name):
    """Force-break the DetachableJoint weld (same topic grasp_manager uses)."""
    subprocess.run(["gz", "topic", "-t", f"/grasp/{name}/detach", "-m", "gz.msgs.Empty", "-p", ""],
                   capture_output=True, text=True)


def gz_model_pose(name):
    """(x, y, z), tilt_deg of a model, from one Gazebo pose message."""
    out = subprocess.run(["gz", "topic", "-e", "-t", f"/world/{WORLD}/pose/info", "-n", "1"],
                         capture_output=True, text=True, timeout=15).stdout
    m = re.search(r'name: "' + re.escape(name) + r'"\s*\n\s*id: \d+\s*\n\s*position \{(.*?)\}'
                  r'\s*orientation \{(.*?)\}', out, re.S)
    if not m:
        return None, None

    def get(block, k):  # protobuf text omits fields that are exactly 0
        mm = re.search(rf"\b{k}: ([-+\d.eE]+)", block)
        return float(mm.group(1)) if mm else 0.0
    p, q = m.group(1), m.group(2)
    xyz = (get(p, "x"), get(p, "y"), get(p, "z"))
    qx, qy = get(q, "x"), get(q, "y")
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy)))))
    return xyz, tilt


def in_bin(xyz, bin_id):
    bx, by = BINS[bin_id]
    return abs(xyz[0] - bx) <= BIN_HALF and abs(xyz[1] - by) <= BIN_HALF and 0.40 <= xyz[2] <= 0.50


def classify(planner_status, exec_failed, xyz, bin_id):
    if planner_status != "complete":
        return "exec_fail" if exec_failed else "plan_fail"
    if xyz is None:
        return "no_pose"
    if xyz[2] > 0.55:
        return "stuck"           # still hanging from the gripper
    if xyz[2] < 0.30:
        return "flung"           # knocked / flung off the table
    if in_bin(xyz, bin_id):
        return "ok"
    if any(in_bin(xyz, b) for b in BINS):
        return "wrong_bin"
    return "missed"              # on the table but not in a bin


def fmt(xyz):
    return "None" if xyz is None else f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})"


class Runner(Node):
    def __init__(self):
        super().__init__("goal_runner")
        self.pub = self.create_publisher(String, "/planning_goal", 10)
        self.create_subscription(Log, "/rosout", self._on_log, 100)
        self.result = None
        self.detail = []
        self.planner_has_world = False
        self.gripper_ready = False

    def _on_log(self, msg):
        if "grasp_manager" in msg.name and "startup: detached" in msg.msg:
            self.gripper_ready = True
        if "execution_status FAIL" in msg.msg:
            self.detail.append("exec:" + msg.msg.split("execution_status")[-1].strip())
        if "tamp_planner" not in msg.name:
            return
        if "got /world_state" in msg.msg:
            self.planner_has_world = True
        elif "PLAN COMPLETE" in msg.msg:
            self.result = "complete"
        elif "TAMP FAILED" in msg.msg:
            self.result = "failed"
            i = msg.msg.find("falsified=")
            self.detail.append("plan:" + (msg.msg[i:] if i >= 0 else msg.msg[-80:]))
        elif "no object of class" in msg.msg:
            self.result = "failed"
            self.detail.append("plan:no_object")

    def spin_for(self, sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def wait_ready(self):
        print("waiting for the planner to be ready ...", flush=True)
        while self.pub.get_subscription_count() == 0 or not self.planner_has_world:
            rclpy.spin_once(self, timeout_sec=0.2)
        t0 = time.time()
        while not self.gripper_ready and time.time() - t0 < 40.0:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.spin_for(5.0)
        print("planner ready.\n", flush=True)

    def reset(self):
        for name in HOME:
            gz_detach(name)
        self.spin_for(0.5)
        for name, xyz in HOME.items():
            if not gz_set_pose(name, xyz):
                print(f"    WARNING: set_pose failed for {name}", flush=True)
        self.spin_for(4.0)  # let items settle and the planner see the new poses

    def run_goal(self, item, bin_id):
        self.result = None
        self.detail = []
        msg = String()
        msg.data = json.dumps({"type": "place_class", "class": item, "surface": bin_id})
        self.pub.publish(msg)
        t0 = time.time()
        while self.result is None and time.time() - t0 < TIMEOUT:
            rclpy.spin_once(self, timeout_sec=0.2)
        status = self.result or "timeout"
        self.spin_for(3.0)  # let the released item settle
        xyz, _ = gz_model_pose(item)
        exec_failed = any(d.startswith("exec:") for d in self.detail)
        return classify(status, exec_failed, xyz, bin_id), xyz, time.time() - t0, list(self.detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3, help="episodes per (item, bin) pair")
    ap.add_argument("--items", default=",".join(HOME), help="comma list, e.g. paper or can,bottle")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    items = [i for i in args.items.split(",") if i]

    episodes = [(i, b) for i in items for b in BINS] * args.trials
    random.Random(args.seed).shuffle(episodes)

    rclpy.init()
    node = Runner()
    node.wait_ready()
    per_pair = defaultdict(list)
    kinds = Counter()
    ok_times = []
    for n, (item, bin_id) in enumerate(episodes, 1):
        node.reset()
        start, tilt = node.gz_start = gz_model_pose(item)
        kind, xyz, dt, detail = node.run_goal(item, bin_id)
        per_pair[(item, bin_id)].append(kind == "ok")
        kinds[kind] += 1
        if kind == "ok":
            ok_times.append(dt)
        tilt_s = "?" if tilt is None else f"{tilt:.1f}deg"
        print(f"[{n:2d}/{len(episodes)}] {item:6s} -> {bin_id:10s}: {kind:9s} "
              f"start={fmt(start)} tilt={tilt_s}  final={fmt(xyz)}  ({dt:.0f}s)", flush=True)
        for d in detail:
            print(f"        {d}", flush=True)

    total_ok = kinds["ok"]
    print(f"\nsuccess rate: {total_ok}/{len(episodes)} = {100 * total_ok / len(episodes):.0f}%")
    if ok_times:
        print(f"mean time of successful episodes: {sum(ok_times) / len(ok_times):.0f}s")
    print("failure kinds: " + ", ".join(f"{k}={v}" for k, v in kinds.items() if k != "ok"))
    print("\n          " + "  ".join(f"{b:10s}" for b in BINS))
    for item in items:
        cells = [f"{sum(per_pair[(item, b)])}/{len(per_pair[(item, b)])}" for b in BINS]
        print(f"  {item:7s} " + "  ".join(f"{c:10s}" for c in cells))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
