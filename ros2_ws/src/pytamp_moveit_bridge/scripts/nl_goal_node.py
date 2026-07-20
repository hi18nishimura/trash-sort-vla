#!/usr/bin/env python3
"""Natural-language goal node — the flagship's ROS front door.

Subscribes ``/command`` (std_msgs/String, an English instruction) and ``/world_state``
(the live perceived scene), grounds the command to a validated PDDL goal spec with
``tamp.nl.GoalGrounder`` (Claude API / NVIDIA / Ollama / offline grammar — whatever
is available), and publishes it as JSON on ``/planning_goal`` for ``tamp_planner_node``
to plan + execute. Also republishes a short human-readable line on ``/plan_explanation``.

Runs under the venv python (needs the ``tamp`` package). The grounder is
LLM-optional, so this works with no API key (offline grammar) or with one.
"""
import json
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from pytamp_moveit_bridge.msg import WorldState

from tamp.nl import GoalGrounder
from tamp.ros_conversions import world_state_from_ros


class NLGoalNode(Node):
    def __init__(self):
        super().__init__("nl_goal")
        self.declare_parameter("use_llm", True)   # False -> force offline grammar
        self.grounder = GoalGrounder(use_llm=self.get_parameter("use_llm").value)
        self.latest_world = None
        self.create_subscription(WorldState, "/world_state", self._on_world, 10)
        self.create_subscription(String, "/command", self._on_command, 10)
        self.goal_pub = self.create_publisher(String, "/planning_goal", 10)
        self.expl_pub = self.create_publisher(String, "/plan_explanation", 10)
        backends = [n for n, _ in self.grounder._backends] or ["offline grammar"]
        self.get_logger().info(f"nl_goal ready — grounding via: {backends}")

    def _on_world(self, msg):
        self.latest_world = world_state_from_ros(msg)

    def _on_command(self, msg):
        cmd = msg.data
        self.get_logger().info(f'command: "{cmd}"')
        if self.latest_world is None:
            self._explain("I don't have a world state yet — is perception running?")
            return
        r = self.grounder.ground(cmd, self.latest_world)
        if not r.success:
            self._explain(f"I couldn't turn that into a goal: {r.error}")
            return
        self.get_logger().info(f"grounded [{r.source}] -> {r.goal}")
        self._explain(f'Understood "{cmd}" as {r.goal} (via {r.source}); planning now.')
        self.goal_pub.publish(String(data=json.dumps(r.goal)))

    def _explain(self, text):
        self.get_logger().info(text)
        self.expl_pub.publish(String(data=text))


def main():
    rclpy.init(args=sys.argv)
    node = NLGoalNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
