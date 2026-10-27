#!/usr/bin/env python3
"""Drive the Panda in Gazebo with the SmolVLA policy (served by tools/policy_server.py).

Run with the SYSTEM python (ROS 2), after the simulator (item:=none) and the policy server:
  python3 tools/vla_node.py
Start:  ros2 topic pub --once /vla/task std_msgs/msg/String "data: 'put the can in the blue bin'"
Stop:   ros2 topic pub --once /vla/task std_msgs/msg/String "data: 'stop'"

Gripper handling (same weld mechanism the expert used):
  - the gripper only switches after `debounce` consecutive agreeing predictions
  - closing welds the item nearest to the hand (x-y), unless something is already held
  - re-closing before a pending release fires cancels the release (keep holding)
"""
import re
import subprocess
import time
from multiprocessing.connection import Client

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

WORLD = "tabletop"
ARM = [f"panda_joint{i}" for i in range(1, 8)]
FINGERS = ["panda_finger_joint1", "panda_finger_joint2"]
ITEMS = {"can": "can_1", "bottle": "bottle_1", "paper": "paper_1"}   # gz model -> object id
OPEN_POS, CLOSE_POS = 0.042, 0.022


def dur(sec):
    d = DurationMsg()
    d.sec = int(sec)
    d.nanosec = int((sec - int(sec)) * 1e9)
    return d


def gz_item_positions():
    """{model: (x, y, z)} for all items, from ONE Gazebo pose message."""
    out = subprocess.run(["gz", "topic", "-e", "-t", f"/world/{WORLD}/pose/info", "-n", "1"],
                         capture_output=True, text=True, timeout=10).stdout
    res = {}
    for name in ITEMS:
        m = re.search(r'name: "' + name + r'"\s*\n\s*id: \d+\s*\n\s*position \{(.*?)\}', out, re.S)
        if m:
            def get(k, block=m.group(1)):
                mm = re.search(rf"\b{k}: ([-+\d.eE]+)", block)
                return float(mm.group(1)) if mm else 0.0
            res[name] = (get("x"), get("y"), get("z"))
    return res


class VlaNode(Node):
    def __init__(self, port, rate, lookahead, debounce):
        super().__init__("vla_node")
        self.conn = Client(("127.0.0.1", port), authkey=b"trash-vla")
        self.lookahead = lookahead
        self.debounce = debounce
        self.img = None
        self.q = None
        self.task = None
        self.gripper_open = True
        self.held = None
        self.pending = None          # (time, "grasp" | "release")
        self.grip_votes = 0          # consecutive predictions that disagree with the gripper state
        self.steps = 0
        self.create_subscription(Image, "/overhead_camera/image", self._on_img, 2)
        self.create_subscription(JointState, "/joint_states", self._on_js, 20)
        self.create_subscription(String, "/vla/task", self._on_task, 10)
        self.arm_pub = self.create_publisher(JointTrajectory, "/panda_arm_controller/joint_trajectory", 10)
        self.hand_pub = self.create_publisher(JointTrajectory, "/panda_hand_controller/joint_trajectory", 10)
        self.grasp_pub = self.create_publisher(String, "/grasp_cmd", 10)
        self.status_pub = self.create_publisher(String, "/vla/status", 10)
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        self.create_timer(1.0 / rate, self._step)
        self.get_logger().info(f"VLA node ready (debounce={debounce}); waiting for /vla/task")

    # ---------------- inputs ----------------
    def _on_img(self, msg):
        if msg.encoding == "rgb8":
            self.img = (bytes(msg.data), msg.height, msg.width)
        elif msg.encoding == "bgr8":
            a = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)[..., ::-1]
            self.img = (a.tobytes(), msg.height, msg.width)

    def _on_js(self, msg):
        d = dict(zip(msg.name, msg.position))
        if all(j in d for j in ARM) and "panda_finger_joint1" in d:
            self.q = [float(d[j]) for j in ARM] + [float(d["panda_finger_joint1"])]

    def _on_task(self, msg):
        text = msg.data.strip()
        if text == "stop":
            self.task = None
            self.pending = None
            self._status("stopped")
            return
        self.conn.send({"cmd": "reset"})
        self.conn.recv()
        self.task = text
        self.gripper_open = True
        self.held = None
        self.pending = None
        self.grip_votes = 0
        self.steps = 0
        self._send_hand(OPEN_POS)
        self._status(f"started: {text}")

    # ---------------- outputs ----------------
    def _status(self, text):
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def _send_arm(self, pos):
        jt = JointTrajectory()
        jt.joint_names = ARM
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in pos]
        p.time_from_start = dur(self.lookahead)
        jt.points = [p]
        self.arm_pub.publish(jt)

    def _send_hand(self, pos):
        jt = JointTrajectory()
        jt.joint_names = FINGERS
        p = JointTrajectoryPoint()
        p.positions = [pos, pos]
        p.time_from_start = dur(0.5)
        jt.points = [p]
        self.hand_pub.publish(jt)

    def _grasp(self):
        """Weld the item closest (in x-y) to the hand, like the expert's grasp_manager call."""
        if self.held:              # already holding something: never re-weld
            return
        try:
            tr = self.tf.lookup_transform("world", "panda_hand", rclpy.time.Time())
            hx, hy = tr.transform.translation.x, tr.transform.translation.y
        except Exception as exc:
            self._status(f"grasp skipped (no TF: {type(exc).__name__})")
            return
        items = gz_item_positions()
        if not items:
            return
        name, (x, y, z) = min(items.items(), key=lambda kv: (kv[1][0] - hx) ** 2 + (kv[1][1] - hy) ** 2)
        dist = ((x - hx) ** 2 + (y - hy) ** 2) ** 0.5
        if dist > 0.05:
            self._status(f"closed on nothing (nearest {name} is {dist * 100:.1f} cm away)")
            return
        self.held = ITEMS[name]
        self.grasp_pub.publish(String(data=f"grasp:{self.held}:{x:.4f},{y:.4f},{z:.4f}"))
        self._status(f"grasp:{self.held}")

    def _release(self):
        if self.held:
            self.grasp_pub.publish(String(data=f"release:{self.held}"))
            self._status(f"release:{self.held}")
            self.held = None

    # ---------------- control loop ----------------
    def _step(self):
        now = time.time()
        if self.pending and now >= self.pending[0]:
            kind = self.pending[1]
            self.pending = None
            self._grasp() if kind == "grasp" else self._release()
        if self.task is None or self.img is None or self.q is None:
            return
        img, h, w = self.img
        self.conn.send({"cmd": "act", "image": img, "h": h, "w": w, "state": self.q, "task": self.task})
        reply = self.conn.recv()
        a = reply["action"]
        self._send_arm(a[:7])
        want_open = a[7] > 0.5
        self.grip_votes = self.grip_votes + 1 if want_open != self.gripper_open else 0
        if self.grip_votes >= self.debounce:   # switch only after N agreeing predictions
            self.grip_votes = 0
            self.gripper_open = want_open
            self._send_hand(OPEN_POS if want_open else CLOSE_POS)
            if want_open:
                self.pending = (now + 0.8, "release")
            elif self.held:
                self.pending = None    # re-closed before the release fired: keep holding
            else:
                self.pending = (now + 1.0, "grasp")
            self._status("gripper " + ("open" if want_open else "close"))
        self.steps += 1
        if self.steps % 50 == 0:
            self.get_logger().info(f"step {self.steps}, inference {reply['ms']:.0f} ms")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6000)
    ap.add_argument("--rate", type=float, default=10.0, help="control rate [Hz] (= dataset fps)")
    ap.add_argument("--lookahead", type=float, default=0.2, help="time to reach each target [s]")
    ap.add_argument("--debounce", type=int, default=5, help="steps before the gripper may switch")
    args = ap.parse_args()
    rclpy.init()
    node = VlaNode(args.port, args.rate, args.lookahead, args.debounce)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
