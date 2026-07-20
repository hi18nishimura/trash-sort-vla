#!/usr/bin/env python3
"""Mock perception: publishes a /world_state built from a benchmark scene, so the
TAMP pipeline can run without a camera/YOLOv8 (the reference's real perception is
object_state_manager.py). Selects a scene with the `scene` parameter.
"""
import os
import sys

import re
import subprocess

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion

from pytamp_moveit_bridge.msg import ObjectState, WorldState

# Make the repo's `benchmarks`/`tamp` packages importable regardless of how the
# node is launched (the installed script lives under ros2_ws/install/...).
_REPO = os.environ.get("TAMP_REPO")
if not _REPO:
    _d = os.path.dirname(os.path.realpath(__file__))
    while _d != "/":
        if os.path.isdir(os.path.join(_d, "benchmarks")) and os.path.isdir(os.path.join(_d, "tamp")):
            _REPO = _d
            break
        _d = os.path.dirname(_d)
if _REPO and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmarks import scene_stack3, scene_replan, scene_pnp

SCENES = {"stack3": scene_stack3, "replan": scene_replan, "pnp": scene_pnp}


class MockPerception(Node):
    def __init__(self):
        super().__init__("mock_perception")
        self.declare_parameter("scene", "stack3")
        # live_poses: track the objects' REAL Gazebo poses instead of republishing the
        # scene's start positions forever. Static poses are fine for a single pick, but
        # they break any MULTI-step task: once cup_1 is placed on the tray, a static
        # /world_state still claims it is at its start spot, so the executor's planning
        # scene holds a PHANTOM cup where nothing is while the REAL cup on the tray is
        # invisible to OMPL — the arm then sweeps through it and knocks it off (observed:
        # cup_1 ended at (0.341,0.116) after being placed at (0.55,0)). Poses come from
        # the gz pose bridge (/model_poses), i.e. simulator ground truth — this is a
        # ground-truth perception stand-in, NOT a vision system (yolo_perception.py is
        # the real-camera path). Off by default so the ROS-free scenes are unaffected.
        self.declare_parameter("live_poses", False)
        # object_id:gz_model_name pairs, e.g. ["cup_1:cup", "cup_2:cup2"]
        self.declare_parameter("model_map", [""])
        self.declare_parameter("gz_world", "tabletop")
        name = self.get_parameter("scene").value
        self.world = SCENES[name].make_world()
        self.live = bool(self.get_parameter("live_poses").value)
        self.gz_world = self.get_parameter("gz_world").value
        self.model_map = {}
        for pair in self.get_parameter("model_map").value or []:
            if ":" in pair:
                oid, model = pair.split(":", 1)
                self.model_map[model.strip()] = oid.strip()
        self._live_pos = {}          # object_id -> (x, y, z) from Gazebo
        # Publish the scene's known start poses for the first few ticks (see _publish):
        # they are correct at t=0 and immune to the startup weld/settle transient.
        self._settle_ticks = 5
        self.declare_parameter("refresh_every", 4)   # ticks between gz pose queries
        self.refresh_every = int(self.get_parameter("refresh_every").value)
        self._refresh_countdown = 1
        self.pub = self.create_publisher(WorldState, "/world_state", 10)
        self.timer = self.create_timer(1.0, self._publish)
        self.get_logger().info(
            f"mock_perception publishing scene '{name}' ({len(self.world.objects)} objects)"
            f" on /world_state" + (f"; LIVE gz poses for {self.model_map}"
                                   if self.live else "; static scene poses"))

    def _refresh_live_poses(self):
        """Read the objects' REAL poses from Gazebo, by NAME.

        NOTE: we shell out to `gz topic` rather than subscribe to the bridged
        /model_poses. The ros_gz Pose_V->TFMessage bridge publishes every transform with
        an EMPTY child_frame_id, so the model name — the only way to tell cup from cup2 —
        is lost and a subscriber cannot map poses back to objects at all. The gz CLI
        output does carry `name: "..."`. (grasp_manager reads poses the same way.)
        """
        try:
            out = subprocess.run(
                ["gz", "topic", "-e", "-t", f"/world/{self.gz_world}/pose/info", "-n", "1"],
                capture_output=True, text=True, timeout=4).stdout
        except Exception as exc:
            self.get_logger().warn(f"gz pose query failed: {exc}")
            return
        for part in re.split(r'\n\s*name:\s*"', out)[1:]:
            model = part.split('"', 1)[0]
            oid = self.model_map.get(model)
            if not oid:
                continue
            pm = re.search(r'position\s*\{([^}]*)\}', part)
            if not pm:
                continue
            # gz omits zero-valued fields, so a missing axis means 0.0.
            vals = {}
            for ax in ("x", "y", "z"):
                m = re.search(rf'\b{ax}:\s*(-?[0-9.eE+-]+)', pm.group(1))
                vals[ax] = float(m.group(1)) if m else 0.0
            self._live_pos[oid] = (vals["x"], vals["y"], vals["z"])

    def _publish(self):
        if self.live:
            # Don't trust Gazebo until the scene has SETTLED. The DetachableJoint plugin
            # attaches every graspable to the wrist at world load (grasp_manager breaks
            # those welds at startup), so sampling too early reports the cups stuck to
            # the arm rather than resting on the table — the reasoner then infers no
            # on(cup, workspace) and the planner returns UNSOLVABLE before anything moves.
            if self._settle_ticks > 0:
                self._settle_ticks -= 1
            else:
                # Refresh only every Nth tick: each refresh spawns a `gz topic` CLI
                # process (a full gz-transport node), which is wasteful at 1 Hz.
                # Objects only move when we move them, so seconds of staleness are
                # harmless. (An earlier comment here blamed this polling for planner
                # UNSOLVABLE failures — that was WRONG: those were traced to a
                # marginal grasp-IK falsifying `graspable`, see motion_planner.py.)
                self._refresh_countdown -= 1
                if self._refresh_countdown <= 0:
                    self._refresh_countdown = self.refresh_every
                    self._refresh_live_poses()
        msg = WorldState()
        for o in self.world.objects:
            os_msg = ObjectState()
            os_msg.object_id = o.object_id
            os_msg.class_name = o.class_name
            p = o.pose
            xyz = self._live_pos.get(o.object_id)          # real pose if we have one
            if xyz is None:
                xyz = (p.position[0], p.position[1], p.position[2])
            os_msg.pose = Pose(
                position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
                orientation=Quaternion(w=float(p.orientation[0]), x=float(p.orientation[1]),
                                       y=float(p.orientation[2]), z=float(p.orientation[3])))
            os_msg.dimensions = Vector3(x=float(o.dimensions[0]), y=float(o.dimensions[1]), z=float(o.dimensions[2]))
            os_msg.confidence = float(o.confidence)
            msg.objects.append(os_msg)
        self.pub.publish(msg)


def main():
    rclpy.init(args=sys.argv)
    node = MockPerception()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
