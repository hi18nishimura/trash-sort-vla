#!/usr/bin/env python3
"""Minimal /task_plan consumer: logs each refined TaskAction as it arrives.

Stands in for the reference's action_executor.py (MoveItPy pick/place) when
running the planning pipeline without a full MoveIt/controller stack. Proves the
end-to-end pub/sub chain: perception -> planner -> execution interface.
"""
import sys

import rclpy
from rclpy.node import Node

from pytamp_moveit_bridge.msg import TaskAction


class ActionLogger(Node):
    def __init__(self):
        super().__init__("action_logger")
        self.create_subscription(TaskAction, "/task_plan", self._on_action, 10)
        self.n = 0
        self.get_logger().info("action_logger ready — listening on /task_plan")

    def _on_action(self, msg):
        self.n += 1
        p = msg.target_pose.position
        self.get_logger().info(
            f"[step {self.n}] {msg.action_type} {msg.object_id} -> "
            f"target=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")


def main():
    rclpy.init(args=sys.argv)
    node = ActionLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
