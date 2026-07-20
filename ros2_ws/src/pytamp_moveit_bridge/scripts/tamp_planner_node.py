#!/usr/bin/env python3
"""Real TAMP planner node — replaces the reference's stubbed planner_node.py
(MCTS placeholder) and scene_interface.py (commented-out PyTAMP).

Subscribes /world_state and a /planning_goal trigger, runs the full PDDL+PyTAMP
TAMP loop (tamp.tamp_solver.TampSolver), and publishes the refined,
geometrically-feasible plan on /task_plan (consumed by the executor / logger).

Requires the `tamp` core on PYTHONPATH (export PYTHONPATH=$PYTHONPATH:<repo root>)
and its deps (pykin, trimesh, numpy<2, unified-planning[fast-downward]).
"""
import json
import os
import sys

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion

from pytamp_moveit_bridge.msg import TaskAction, WorldState

from tamp.tamp_solver import TampSolver
from tamp.ros_conversions import (surfaces_from_config, task_action_to_ros,
                                  world_state_from_ros)
from tamp.execution.feedback import parse_status, predicate_for_action


class TampPlannerNode(Node):
    def __init__(self):
        super().__init__("tamp_planner")
        self.declare_parameter("scene_config", "")
        self.declare_parameter("max_replans", 8)
        self.declare_parameter("default_surface", "")  # objects w/o inferred support rest here
        self.declare_parameter("max_exec_replans", 5)  # cap on execution-failure replans

        cfg_path = self.get_parameter("scene_config").value
        self.surfaces = []
        if cfg_path and os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                self.surfaces = surfaces_from_config(yaml.safe_load(fh))
            self.get_logger().info(f"loaded {len(self.surfaces)} surface(s) from {cfg_path}")

        self.get_logger().info("initializing TampSolver (loading pykin robot model)...")
        ds = self.get_parameter("default_surface").value or None
        self.solver = TampSolver(max_replans=int(self.get_parameter("max_replans").value),
                                 default_surface=ds)

        self.max_exec_replans = int(self.get_parameter("max_exec_replans").value)
        self.latest_world = None
        # closed-loop dispatch state (A2): one step out at a time, replan on failure
        self.plan = []
        self.step_i = 0
        self.current_goal = None
        self.exec_falsified = []   # bridge predicates learned from execution failures
        self.exec_replans = 0
        self.create_subscription(WorldState, "/world_state", self._on_world, 10)
        self.create_subscription(String, "/planning_goal", self._on_goal, 10)
        self.create_subscription(String, "/execution_status", self._on_status, 10)
        self.plan_pub = self.create_publisher(TaskAction, "/task_plan", 10)
        self.get_logger().info("TAMP planner ready — waiting for /world_state + /planning_goal")

    def _on_world(self, msg):
        self.latest_world = world_state_from_ros(msg, surfaces=self.surfaces)
        self.get_logger().info(f"got /world_state: {len(msg.objects)} object(s)")

    def _on_goal(self, msg):
        if self.latest_world is None:
            self.get_logger().warn("goal received but no /world_state yet")
            return
        try:
            goal = json.loads(msg.data) if msg.data.lstrip().startswith("{") else yaml.safe_load(msg.data)
        except Exception as exc:
            self.get_logger().error(f"bad goal spec: {exc}")
            return

        # Resolve a class-based goal ("put a cup on the tray") to a concrete
        # detected object id from the latest YOLO /world_state.
        if goal.get("type") == "place_class" and "_resolved" not in goal:
            match = next((o.object_id for o in self.latest_world.objects
                          if o.class_name == goal["class"]), None)
            if match is None:
                self.get_logger().error(f"no object of class '{goal['class']}' detected")
                return
            goal = {**goal, "_resolved": match}
            self.get_logger().info(f"resolved {goal['class']} -> {match}")

        # New goal: reset the execution-feedback state and (re)plan from scratch.
        self.current_goal = goal
        self.exec_falsified = []
        self.exec_replans = 0
        self._solve_and_dispatch()

    def _solve_and_dispatch(self):
        """Solve (seeding any execution-learned falsifications) and start dispatch."""
        self.get_logger().info(f"planning for goal: {self.current_goal}"
                               + (f"  (exec-falsified: {self.exec_falsified})" if self.exec_falsified else ""))
        # Log the EXACT world we are planning on. The same goal+scene solves offline but
        # returned UNSOLVABLE here, so the difference has to be the live world state.
        try:
            w = self.latest_world
            self.get_logger().info(
                "  world: " + "; ".join(
                    f"{o.object_id}[{o.class_name}]@({o.pose.position[0]:.3f},"
                    f"{o.pose.position[1]:.3f},{o.pose.position[2]:.3f})"
                    f"dim{tuple(round(float(d),3) for d in o.dimensions)}" for o in w.objects)
                + " | surfaces: " + "; ".join(
                    f"{s.surface_id}[{s.class_name}]@({s.pose.position[0]:.3f},"
                    f"{s.pose.position[1]:.3f},{s.pose.position[2]:.3f})"
                    f"dim{tuple(round(float(d),3) for d in s.dimensions)}" for s in w.surfaces))
            from tamp.reasoning.object_reasoner import infer_relations
            rel = infer_relations(w)
            self.get_logger().info(
                f"  inferred: support_of={rel.support_of} clear={sorted(rel.clear)} "
                f"| default_surface={self.solver.default_surface}")
        except Exception as exc:
            self.get_logger().warn(f"  (could not dump world: {exc})")
        result = self.solver.solve(self.latest_world, self.current_goal,
                                   pre_falsified=self.exec_falsified)
        if not result.success:
            # ALWAYS show which bridge predicates were falsified. Without this, a
            # geometric-refiner falsification (e.g. graspable(cup_2) after a marginal
            # IK failure) makes the replan genuinely unsolvable and the only visible
            # symptom is "symbolic planning failed (UNSOLVABLE_INCOMPLETELY)" — which
            # reads as a PDDL bug and cost days of misdirected debugging.
            self.get_logger().error(
                f"TAMP FAILED: {result.reason} (replans={result.replans}, "
                f"falsified={result.falsified})")
            self.plan = []
            return
        self.get_logger().info(
            f"TAMP SOLVED in {result.replans} replan(s); falsified={result.falsified}; "
            f"dispatching {len(result.plan)} step(s) one at a time on /task_plan")
        self.plan = result.plan
        self.step_i = 0
        self._dispatch_next()

    def _dispatch_next(self):
        """Publish the current step; the executor reports back on /execution_status."""
        if self.step_i >= len(self.plan):
            self.get_logger().info("PLAN COMPLETE — all steps executed")
            return
        step = self.plan[self.step_i]
        self.get_logger().info(f"-> dispatch step {self.step_i + 1}/{len(self.plan)}: "
                               f"{step.action_type} {step.object_id}")
        self.plan_pub.publish(task_action_to_ros(step, TaskAction, Pose, Point, Quaternion))

    def _on_status(self, msg):
        """Close the loop: advance on success, replan (falsifying the offending
        bridge predicate) on an execution failure."""
        st = parse_status(msg.data)
        if st is None or not self.plan or self.step_i >= len(self.plan):
            return
        # The status MUST belong to the step we are waiting on. The wire format is
        # keyed by (action_type, object_id) precisely so we can check this — but the
        # check was missing, so a late status from a superseded plan (e.g. a new
        # /planning_goal arrived mid-step) would advance the NEW plan's index,
        # silently skipping its step 0 while still reporting "PLAN COMPLETE".
        expected = self.plan[self.step_i]
        if (st.action_type, st.object_id) != (expected.action_type, expected.object_id):
            self.get_logger().warn(
                f"ignoring stale /execution_status {st.action_type}:{st.object_id} "
                f"(waiting on {expected.action_type}:{expected.object_id})")
            return
        if st.ok:
            self.step_i += 1
            self._dispatch_next()
            return
        # failure: map it to a bridge predicate and re-solve
        action = expected
        pred = predicate_for_action(action, st.reason)
        self.get_logger().warn(f"step {self.step_i + 1} FAILED ({st.reason}); "
                               f"falsifying {pred} and replanning")
        if pred and pred not in self.exec_falsified:
            self.exec_falsified.append(pred)
        self.exec_replans += 1
        if self.exec_replans > self.max_exec_replans:
            self.get_logger().error(f"giving up after {self.exec_replans} execution replans")
            self.plan = []
            return
        self._solve_and_dispatch()


def main():
    # Route the tamp core's Python logging (tamp.solver / tamp.planner INFO lines,
    # e.g. "replan: falsifying ('graspable','cup_2')") to stdout so they appear in
    # the launch log. Without a handler these messages are silently dropped and the
    # closed loop's most important diagnostic — WHY it replanned — is invisible.
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="[tamp] %(name)s: %(message)s", stream=sys.stdout)
    rclpy.init(args=sys.argv)
    node = TampPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
