#!/usr/bin/env python3
"""Kinematic grasp-attach for Gazebo pick-and-place.

Gazebo friction-grasping a box with the Panda is unstable, so we attach the
object kinematically: on "grasp:<obj>" we capture the object's pose relative to
the gripper (panda_hand) and thereafter set the object's pose each tick to follow
the gripper via the gz `/world/<world>/set_pose` service (UserCommands system).
On "release:<obj>" we stop following and physics takes over (the object rests on
whatever is below it).

Runs under SYSTEM python3 (needs rclpy + tf2_ros; no ultralytics). Object world
poses come from the gz pose feedback bridged to /model_poses (tf2_msgs/TFMessage).
"""
import re
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import tf2_ros


def quat_to_R(x, y, z, w):
    n = (x*x+y*y+z*z+w*w) ** 0.5 or 1.0
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = 0.5/np.sqrt(t+1); w = 0.25/s
        x = (R[2,1]-R[1,2])*s; y = (R[0,2]-R[2,0])*s; z = (R[1,0]-R[0,1])*s
    else:
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            s = 2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]); w=(R[2,1]-R[1,2])/s
            x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
        elif i == 1:
            s = 2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2]); w=(R[0,2]-R[2,0])/s
            x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
        else:
            s = 2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1]); w=(R[1,0]-R[0,1])/s
            x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    return x, y, z, w


def mat(p, q):
    T = np.eye(4); T[:3, :3] = quat_to_R(*q); T[:3, 3] = p
    return T


class GraspManager(Node):
    def __init__(self):
        super().__init__("grasp_manager")
        self.declare_parameter("world", "tabletop")
        self.declare_parameter("gripper_frame", "panda_hand")
        self.declare_parameter("base_frame", "world")
        self.world = self.get_parameter("world").value
        self.gripper_frame = self.get_parameter("gripper_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        self.tf_buffer = tf2_ros.Buffer()
        # node=None => the listener gets its OWN node + spin thread, fully independent
        # of this node's executor. Passing `self` (even with spin_thread=True) leaves the
        # /tf subscriptions in this node's default MutuallyExclusive group, so they are
        # serialized behind _on_cmd / _follow — both of which block for seconds in gz
        # subprocesses. The TF buffer then freezes and the object is carried using a
        # stale gripper transform. This is the real fix for that starvation.
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, None, spin_thread=True)
        self.attached = {}      # gz model name -> T_grip_obj (offset)

        # Separate callback groups so the blocking gz subprocesses in the follow timer
        # and the command handler cannot starve each other.
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        self._timer_cb = MutuallyExclusiveCallbackGroup()
        self._cmd_cb = MutuallyExclusiveCallbackGroup()
        self.create_subscription(String, "/grasp_cmd", self._on_cmd, 10,
                                 callback_group=self._cmd_cb)
        # Feedback channel (was ENTIRELY ABSENT): the executor must be able to learn
        # that an attach failed. Without this, a grasp that silently attached nothing
        # was still reported as a completed pick-and-place.
        self.status_pub = self.create_publisher(String, "/grasp_status", 10)

        # --- finger state: a grasp must actually involve the fingers ---------------
        # The attach previously read NO finger position, effort or contact — the object
        # would have followed the hand identically with the gripper wide open. We now
        # require the fingers to have closed before an attach is allowed.
        self.declare_parameter("tcp_z", 0.10)          # fingertip offset along hand +z
        self.declare_parameter("grasp_span", 0.05)     # object must be this near the TCP
        # A grasp = the fingers stopped ON the object, i.e. the joint settled inside a
        # CONTACT BAND, not just "below a threshold". bullet-featherstone does not let
        # the fingers penetrate the cup the way DART did (DART sank to 0.024; bullet
        # stops at true contact ~0.035 for the 5 cm cup). And in free space the fingers
        # close all the way to 0.0 — so "finger < 0.035" both rejected real bullet
        # grasps AND would accept an empty close. The band requires the fingers to have
        # closed in from open (< finger_contact_max, below the 0.038 open width) yet
        # stopped short of full closure (> finger_contact_min), which only happens when
        # something is between them. tcp_err + the drop detector confirm it is the cup.
        self.declare_parameter("finger_contact_max", 0.0385)
        self.declare_parameter("finger_contact_min", 0.015)
        self.declare_parameter("hold_tol", 0.05)       # max drift before we call it dropped
        self.tcp_z = float(self.get_parameter("tcp_z").value)
        self.grasp_span = float(self.get_parameter("grasp_span").value)
        self.finger_contact_max = float(self.get_parameter("finger_contact_max").value)
        self.finger_contact_min = float(self.get_parameter("finger_contact_min").value)
        self.hold_tol = float(self.get_parameter("hold_tol").value)
        self._verify_after = None      # start verifying only after the joint settles
        self._last_grip = None         # for the "is the arm stationary?" gate
        self._finger_pos = None
        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)

        # Models that have a DetachableJoint declared against panda_hand (see
        # description/panda_gazebo.urdf.xacro). The plugin attaches at world load, so
        # we must detach them once at startup to begin with an empty gripper.
        self.declare_parameter("graspable_models", ["cup", "cup2"])
        self.graspable_models = list(self.get_parameter("graspable_models").value)
        self.create_timer(2.0, self._startup_detach_once, callback_group=self._timer_cb)
        self._did_startup_detach = False
        # 2 Hz drop-detector (NOT a teleport loop — see _follow)
        self.create_timer(0.5, self._follow, callback_group=self._timer_cb)

        self.get_logger().info(f"grasp_manager ready (world={self.world}); "
                               f"physics-attach via DetachableJoint {self.graspable_models}")

    def _startup_detach_once(self):
        if self._did_startup_detach:
            return
        self._did_startup_detach = True
        self._detach_all_at_startup()

    def _on_joints(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name == "panda_finger_joint1":
                self._finger_pos = float(pos)
                return

    def _fingers_closed(self) -> bool:
        # True only if the fingers stopped ON an object: closed in from the open width
        # but not all the way to nothing (free space closes fully to ~0.0). See the
        # contact-band rationale where the params are declared.
        return (self._finger_pos is not None
                and self.finger_contact_min < self._finger_pos < self.finger_contact_max)

    def _status(self, ok: bool, obj: str, reason: str):
        """Report the outcome of a grasp/release so the executor can close the loop."""
        self.status_pub.publish(String(data=f"{'ok' if ok else 'fail'}:{obj}:{reason}"))
        if not ok:
            self.get_logger().warn(f"/grasp_status FAIL:{obj}:{reason}")

    @staticmethod
    def _floats(body, keys, default=0.0):
        """Parse named floats out of a gz TextFormat block. gz omits zero-valued
        fields, so a missing key means 0. The old pattern `[-\\d.e]+` had no exponent
        sign, so `1e+06` parsed as "1e" -> float("1e") -> ValueError, uncaught, which
        killed the node from inside a callback."""
        num = r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf))'
        out = []
        for k in keys:
            m = re.search(rf'\b{k}:\s*{num}', body)
            v = default
            if m:
                try:
                    v = float(m.group(1))
                except ValueError:
                    v = default
            out.append(v if np.isfinite(v) else default)
        return out

    def _query_model_poses(self):
        """Read model names + FULL world poses (position AND orientation) from gz.

        Orientation matters: the grasp captures the object's true pose relative to the
        gripper, so it keeps its real orientation while carried. The previous version
        parsed only position and then hard-coded the held object upright."""
        try:
            out = subprocess.run(
                ["gz", "topic", "-e", "-t", f"/world/{self.world}/pose/info", "-n", "1"],
                capture_output=True, text=True, timeout=4).stdout
        except Exception as exc:
            self.get_logger().warn(f"gz pose query failed: {exc}")
            return {}
        # gz omits zero-valued fields (e.g. 'y: 0' is not printed), so missing axes
        # default to 0 -- and for a quaternion, w defaults to 1 (identity).
        poses = {}
        for part in re.split(r'\n\s*name:\s*"', out)[1:]:
            name = part.split('"', 1)[0]
            pm = re.search(r'position\s*\{([^}]*)\}', part)
            if not pm:
                continue
            x, y, z = self._floats(pm.group(1), ("x", "y", "z"))
            om = re.search(r'orientation\s*\{([^}]*)\}', part)
            if om:
                qx, qy, qz = self._floats(om.group(1), ("x", "y", "z"), 0.0)
                qw = self._floats(om.group(1), ("w",), 1.0)[0]
            else:
                qx = qy = qz = 0.0
                qw = 1.0
            poses[name] = mat([x, y, z], [qx, qy, qz, qw])   # full 4x4
        return poses

    def _gripper_world(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.gripper_frame, rclpy.time.Time())
            p = tf.transform.translation; q = tf.transform.rotation
            return mat([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
        except Exception:
            return None

    # substrings identifying non-graspable models (robot links + furniture)
    # Non-graspable entities (robot links + furniture + lights). Matched on NAME
    # SEGMENTS, not raw substrings: a substring test wrongly skipped COCO's "handbag"
    # (contains "hand") while *failing* to skip `random` — a dummy link in panda.urdf
    # rigidly attached to panda_hand, which therefore sits within 0.10 m of every grasp
    # target and was a live candidate to be "grasped" instead of the object.
    SKIP_TOKENS = frozenset((
        "panda", "ground", "table", "tray", "rgbd", "camera", "link", "finger",
        "hand", "wall", "sun", "world", "random", "light", "fill", "plane",
    ))

    @classmethod
    def _is_graspable_name(cls, name: str) -> bool:
        segments = re.split(r"[^a-z0-9]+", name.lower())
        return not any(seg in cls.SKIP_TOKENS for seg in segments if seg)

    def _on_cmd(self, msg):
        # "grasp:<id>:<x>,<y>,<z>"  /  "release:<id>"
        parts = msg.data.split(":")
        cmd = parts[0]
        obj = parts[1] if len(parts) > 1 else ""
        g = self._gripper_world()
        if cmd == "grasp":
            # Refuse to grasp while already holding something. Otherwise a failed
            # PLACE (which leaves the object attached) followed by a replan's PICK
            # would attach a SECOND object at the same TCP, and the next release
            # would drop both at one pose.
            if self.attached:
                self._status(False, obj, f"already_holding:{list(self.attached)}")
                return
            if g is None:
                self._status(False, obj, "no_gripper_tf")
                return
            poses = self._query_model_poses()
            cand = {n: T for n, T in poses.items() if self._is_graspable_name(n)}
            if not cand:
                self._status(False, obj, "no_models_from_gz")
                return
            # Pick the gz model matching the COMMANDED object (nearest to its known
            # grasp position), not merely nearest to the gripper — so an imperfect
            # reach can't grab an already-placed object.
            if len(parts) >= 3:
                tgt = np.array([float(v) for v in parts[2].split(",")])
                ref = "object position"
            else:
                tgt = (g @ np.array([0, 0, self.tcp_z, 1.0]))[:3]; ref = "gripper TCP"
            model = min(cand, key=lambda n: np.linalg.norm(cand[n][:3, 3] - tgt))
            dist = float(np.linalg.norm(cand[model][:3, 3] - tgt))
            if dist > 0.12:
                # THE failure the kinematic attach can actually detect. Reporting it
                # (instead of returning silently) is what lets the planner replan
                # instead of declaring a pick-and-place that moved nothing "complete".
                self._status(False, obj, f"no_object_within_reach:{dist:.3f}m")
                return

            # --- the grasp must actually be a grasp ---------------------------------
            # 1) The object must lie between the fingers, i.e. close to the TCP — not
            #    merely "somewhere within 12 cm of where we thought it was". Previously
            #    the object was SNAPPED to the TCP from wherever it really was, which
            #    erased up to 10 cm of reach error and made a missed reach unfailable.
            T_wo = cand[model]                                  # object pose in world
            tcp_w = (g @ np.array([0, 0, self.tcp_z, 1.0]))[:3]  # fingertip point
            tcp_err = float(np.linalg.norm(T_wo[:3, 3] - tcp_w))
            if tcp_err > self.grasp_span:
                self._status(False, obj, f"object_not_between_fingers:{tcp_err:.3f}m")
                return
            # 2) The fingers must actually have closed. No finger state => no grasp.
            if not self._fingers_closed():
                self._status(False, obj, f"fingers_not_closed:{self._finger_pos}")
                return

            # REAL attach: create a fixed joint in the physics engine between panda_hand
            # and the object, at their current relative pose. Gazebo then carries the
            # object rigidly — no teleporting, no stutter, no free-fall between updates,
            # and it keeps its true grasped offset and orientation.
            if not self._joint_attach(model):
                self._status(False, obj, "physics_attach_failed")
                return
            self.attached[model] = np.linalg.inv(g) @ T_wo   # ground truth for the drop check
            self._verify_after = time.time() + 1.0           # let the joint settle first
            self.get_logger().info(
                f"grasped '{model}' (cmd {obj}); tcp_err={tcp_err*1000:.0f}mm, "
                f"fingers={self._finger_pos:.3f} — fixed joint attached (physics carries it)")
            self._status(True, obj, f"attached:{model}")
        elif cmd == "release":
            # Releasing with nothing *tracked* still detaches every joint. Our bookkeeping
            # can disagree with physics (e.g. the executor timed out waiting for the grasp
            # verdict and took the failure path while the joint HAD in fact attached) — and
            # a stale joint means the object hangs off an open gripper in mid-air. Always
            # break the joints; then report.
            if not self.attached:
                for m in self.graspable_models:
                    self._joint_detach(m)
                self._verify_after = None
                self._status(False, obj, "release_with_nothing_held(detached_all)")
                return
            if g is None:
                self._status(False, obj, "no_gripper_tf")
                return
            # REAL release: break the fixed joint. The object is wherever the arm
            # actually carried it and simply falls/settles under gravity. We do NOT
            # snap it to a computed "place pose" (which used to make XY + orientation
            # exact by construction, erasing all reach error so a placement could never
            # be wrong).
            released = list(self.attached.keys())
            for held in released:
                self._joint_detach(held)
            self.attached.clear()
            self._verify_after = None
            self.get_logger().info(f"released {released} (fixed joint broken; physics settles it)")
            self._status(True, obj, f"released:{released[0] if released else ''}")

    def drop_all(self):
        """Detach without repositioning — used when a step FAILS mid-carry, so the
        object doesn't stay welded to the gripper into the next replan."""
        if self.attached:
            dropped = list(self.attached.keys())
            for held in dropped:
                self._joint_detach(held)
            self.attached.clear()
            self.get_logger().warn(f"dropped {dropped} (failure path; joint broken)")

    # ---------------- real (physics) attach via gz DetachableJoint ----------------
    def _gz_pub_empty(self, topic):
        """One-shot Empty publish to a gz topic. Only fires on grasp/release (twice per
        object), so a subprocess here is fine — unlike the old per-tick set_pose loop."""
        try:
            r = subprocess.run(
                ["gz", "topic", "-t", topic, "-m", "gz.msgs.Empty", "-p", ""],
                capture_output=True, text=True, timeout=3)
            return r.returncode == 0
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"gz topic pub {topic} failed: {exc}")
            return False

    def _joint_attach(self, model):
        return self._gz_pub_empty(f"/grasp/{model}/attach")

    def _joint_detach(self, model):
        return self._gz_pub_empty(f"/grasp/{model}/detach")

    def _detach_all_at_startup(self):
        """The DetachableJoint plugin attaches at world load, so start unattached."""
        for m in self.graspable_models:
            self._joint_detach(m)
        self.get_logger().info(f"startup: detached {self.graspable_models} (begin unattached)")

    def _follow(self):
        """Formerly the teleport loop; now a DROP DETECTOR.

        The object is held by a REAL fixed joint, so Gazebo carries it and we must not
        fight that by re-writing its pose (that teleporting — a `gz service` subprocess
        per tick, so ~1 Hz — is exactly what made the object stutter behind the arm and
        free-fall between updates).

        What we DO owe the closed loop is honesty about whether we are still holding it.
        Publishing to an attach topic tells us nothing: `gz topic` exits 0 whether or not
        a joint was created (that is how a run once reported a successful pick while the
        cup never left the table). So we verify against ground truth: the held object must
        remain at the relative pose it was grasped at. If it diverges, we dropped it —
        report it, and the planner falsifies `graspable` and replans."""
        if not self.attached or self._verify_after is None:
            return
        if time.time() < self._verify_after:      # let the joint settle after attach
            return
        g = self._gripper_world()
        if g is None:
            return

        # ONLY verify while the arm is STATIONARY. _query_model_poses() shells out to
        # `gz topic` and takes ~1 s, so the object pose it returns is ~1 s stale while
        # the gripper TF is fresh. During a fast transfer that lag alone is ~90 mm — it
        # measures LATENCY, not slippage, and produced a false "DROPPED" on a pick-and-
        # place that had actually put the cup on the tray. A drop is still caught: if the
        # object is genuinely not attached it will not be at the grasped offset once the
        # arm stops.
        prev = self._last_grip
        self._last_grip = g[:3, 3].copy()
        if prev is None or float(np.linalg.norm(self._last_grip - prev)) > 0.003:
            return                                 # still moving -> measurement unreliable

        poses = self._query_model_poses()
        for held, T_grip_obj in list(self.attached.items()):
            actual = poses.get(held)
            if actual is None:
                continue
            expected = g @ T_grip_obj
            err = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
            if err > self.hold_tol:
                self.get_logger().error(
                    f"DROPPED '{held}': {err*1000:.0f} mm from where it should be "
                    f"(is the joint actually attached?)")
                self.attached.pop(held, None)
                self._joint_detach(held)
                self._status(False, held, f"dropped:{err:.3f}m")
            else:
                self.get_logger().info(f"holding {held} (err {err*1000:.0f} mm)",
                                       throttle_duration_sec=2.0)

    def _set_pose_mat(self, name, T):
        """set_pose from a 4x4 (keeps the object's real orientation)."""
        x, y, z = T[:3, 3]
        qx, qy, qz, qw = R_to_quat(T[:3, :3])
        return self._set_pose(name, x, y, z, qx, qy, qz, qw)

    def _set_pose(self, name, x, y, z, qx, qy, qz, qw):
        req = (f'name: "{name}", position: {{x: {x}, y: {y}, z: {z}}}, '
               f'orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}')
        try:
            r = subprocess.run(
                ["gz", "service", "-s", f"/world/{self.world}/set_pose",
                 "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                 "--timeout", "1500", "--req", req],
                capture_output=True, text=True, timeout=3)
            return "true" in r.stdout
        except Exception as exc:
            self.get_logger().warn(f"set_pose error: {exc}", throttle_duration_sec=1.0)
            return False


def main():
    from rclpy.executors import MultiThreadedExecutor
    rclpy.init(args=sys.argv)
    node = GraspManager()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
