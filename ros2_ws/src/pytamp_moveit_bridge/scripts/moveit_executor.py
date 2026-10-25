#!/usr/bin/env python3
"""MoveIt 2 executor: plans (OMPL) to each /task_plan target pose using MoveItPy.

This is the real execution layer (adapted from the reference action_executor.py).
For each TaskAction it sets a pose goal for the `panda_arm` group in the
panda_link0 frame and runs the configured planning pipeline (OMPL), reporting
plan success and trajectory length. If a controller manager is available the
trajectory is also executed; otherwise planning alone is the validation.

Launch with the MoveIt config parameters loaded — see tamp_moveit.launch.py.
Runs under system python3 (MoveItPy is a ROS package); it does NOT import the
`tamp` geometry deps, so no venv is required for this node.
"""
import sys

import math
import os
import time

import rclpy
import tf2_ros
import yaml
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, DisplayTrajectory
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String

from pytamp_moveit_bridge.msg import TaskAction, WorldState

from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.robot_state import RobotState

# Collision-free Panda "ready" configuration (the zero config self-collides on
# this URDF: panda_hand-panda_link5 / panda_link5-panda_link7).
PANDA_READY = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


class MoveItExecutor(Node):
    # IMPORTANT: this subscriber node opts out of global ROS args so the launch
    # `__node:=moveit_py` remap (which scopes the MoveIt config params to the
    # MoveItPy node) does not rename / clobber it.
    def __init__(self, robot):
        super().__init__("tamp_moveit_subscriber", use_global_arguments=False)
        # This node opts out of global ROS args (to avoid the __node remap), so it
        # does NOT receive launch params scoped to "moveit_py". Config therefore
        # comes from environment variables (set by the launch / run.sh).
        self.execute = os.environ.get("TAMP_EXECUTE", "0") == "1"
        self.start_from_current = os.environ.get("TAMP_START_CURRENT", "0") == "1"
        self.pickplace = os.environ.get("TAMP_PICKPLACE", "0") == "1"  # full pick/place seq
        self.approach_z = 0.10   # pre-grasp / pre-place height above target (m)
        self.lift_z = 0.15       # lift / retreat height (m)
        self.robot = robot
        self.arm = robot.get_planning_component("panda_arm")
        self.model = robot.get_robot_model()

        # Plan-request parameters (plan() otherwise uses an empty pipeline name).
        self.plan_params = PlanRequestParameters(robot, "ompl")
        self.plan_params.planning_pipeline = "ompl"
        self.plan_params.planning_time = 5.0
        self.plan_params.planning_attempts = 10
        self.plan_params.max_velocity_scaling_factor = float(os.environ.get("TAMP_VEL_SCALE", "1.0"))
        self.plan_params.max_acceleration_scaling_factor = float(os.environ.get("TAMP_VEL_SCALE", "1.0"))

        self.dwell = float(os.environ.get("TAMP_DWELL", "3.0"))  # RViz dwell when not executing
        # RViz animates trajectories published here (MotionPlanning display).
        self.display_pub = self.create_publisher(DisplayTrajectory, "/display_planned_path", 10)
        # Direct controller command (reliable in Gazebo; avoids MoveItPy.execute()
        # deadlocking inside a single-threaded subscriber callback).
        # The arm's FollowJointTrajectory action client is created below (it needs the
        # reentrant callback group). It replaced a publisher on the best-effort
        # /panda_arm_controller/joint_trajectory TOPIC: exactly as with the gripper, the
        # topic is fire-and-forget, so a trajectory could be dropped or silently
        # truncated and the arm would stop centimetres short — the executor then logged
        # "DID NOT ARRIVE: best error 30/90 mm". The action has an accept/execute
        # handshake and a completion result.
        self.grasp_pub = self.create_publisher(String, "/grasp_cmd", 10)
        # Per-step outcome for the closed loop (A2): the planner replans on a fail.
        # Wire format mirrors tamp.execution.feedback (we build it inline so this
        # system-python node needs no tamp import): "<ok|fail>:<act>:<oid>:<reason>".
        self.status_pub = self.create_publisher(String, "/execution_status", 10)
        # Gripper: commanded via the FollowJointTrajectory ACTION, not the
        # /joint_trajectory topic. The topic is best-effort/fire-and-forget — the JTC
        # reliably ran the FIRST goal but silently dropped later ones, so the 2nd/3rd
        # grasp's close barely moved (0.04 -> 0.0377) and failed the closure gate. The
        # action has an accept/execute handshake and does not drop goals. The client is
        # created below in the reentrant _io_cb group so its result callbacks are
        # serviced while _on_action blocks. Both fingers are commanded (see _gripper);
        # joint2 is no longer a mimic because DART cannot honor mimic constraints.
        # panda_finger_joint1/2: 0.04 = open, ~0.022 = closed on a 0.05 m object.
        # OPEN is 0.038, NOT the 0.04 joint limit: when the finger settles exactly on
        # its upper limit (0.0400), Gazebo pins it against the limit constraint and the
        # NEXT close command is silently ignored — the controller reports the goal
        # "successfully reached" while the joint never moves (the 2nd grasp's close
        # failed this way every time; the 1st worked only because its open happened to
        # stop at 0.0386, short of the limit). 0.038 keeps a 2 mm margin off the limit
        # yet still opens the jaw to 7.6 cm — plenty for the 5 cm cup. Must stay above
        # the 0.035 finger_closed gate so an open hand never reads as closed.
        # OPEN 0.042: bullet inflates convex collisions with a contact margin measured at
        # ~8.5 mm per surface on this rig (closes register "contact" at joint 0.033-0.035
        # = 8.5 mm before touching the 50 mm cup). At open 0.038 (76 mm gap), a lateral
        # landing offset of just ~4 mm puts the NEAR finger inside the margin field of
        # the cup and the contact force pushes it OUTWARD while the far finger closes -
        # the captured see-saw (f1 0.038->0.044 while f2 closed) that killed grasps. At
        # 0.042 (84 mm gap) with re-centring to <=6 mm, per-side clearance is >=11 mm >
        # margin. Verified in isolation (no cup): open/close track perfectly, so the
        # controller itself is sound. Stays 3 mm under the 0.045 physics limit (no pin).
        self.GRIPPER_OPEN = 0.042
        # CLOSE target 0.022. NOTE: a full-squeeze (0.0) was tried to force the fingers
        # to true contact, but it made no difference (bullet stalls the fingers at real
        # cup contact ~0.035 regardless of the target) AND it introduced dropped grasps
        # in a batch — the controller keeps commanding 0.0 after the weld attaches,
        # squeezing the light cup out. 0.022 (just inside the cup) is gentle and was the
        # value that first completed the two-cup demo.
        self.GRIPPER_CLOSE = 0.022

        # --- arrival verification (C-3) ---
        self.reach_tol = float(os.environ.get("TAMP_REACH_TOL", "0.02"))   # 2 cm
        self.settle_timeout = float(os.environ.get("TAMP_SETTLE_TIMEOUT", "5.0"))
        self.rtf_slack = float(os.environ.get("TAMP_RTF_SLACK", "4.0"))    # sim may run <1x
        self.tf_buffer = tf2_ros.Buffer()
        # node=None => TransformListener creates its OWN node + spin thread.
        # Passing `self` (even with spin_thread=True) puts the /tf subscriptions in THIS
        # node's default MutuallyExclusive callback group — the same group as the
        # blocking /task_plan callback — so TF is serialized behind us and FREEZES while
        # we poll for arrival. That made the arrival check read the arm's *start* pose
        # forever and report a false "DID NOT ARRIVE" even though the hand was 1 mm from
        # the goal. A dedicated node keeps TF live while we block.
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, None, spin_thread=True)

        # --- grasp feedback (C-2): learn that an attach actually failed ---
        # /grasp_status and /world_state MUST live in a different callback group from
        # /task_plan. _on_action blocks for the whole pick-and-place, and a
        # MutuallyExclusiveCallbackGroup serializes everything in it — so with all three
        # in the node's default group, the grasp verdict could never be delivered while
        # we were waiting for it, and every grasp timed out as "no_grasp_status" even
        # when grasp_manager had already reported success. Needs MultiThreadedExecutor.
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
        self._action_cb = MutuallyExclusiveCallbackGroup()   # long, blocking
        self._io_cb = ReentrantCallbackGroup()               # short, must stay live
        # Gripper + arm action clients (reentrant group: their goal/result callbacks must
        # be serviced while the blocking _on_action pick-and-place is in progress).
        self.gripper_ac = ActionClient(
            self, FollowJointTrajectory,
            "/panda_hand_controller/follow_joint_trajectory",
            callback_group=self._io_cb)
        self.arm_ac = ActionClient(
            self, FollowJointTrajectory,
            "/panda_arm_controller/follow_joint_trajectory",
            callback_group=self._io_cb)
        self._grasp_status = None
        self.create_subscription(String, "/grasp_status", self._on_grasp_status, 10,
                                 callback_group=self._io_cb)
        # Finger state: the gripper must WAIT for the fingers, not for a wall clock.
        self._finger = None
        self.create_subscription(JointState, "/joint_states", self._on_joints, 10,
                                 callback_group=self._io_cb)

        # --- PLANNING SCENE: give OMPL something to avoid ---------------------
        # Previously nothing was ever fed to the PlanningSceneMonitor, so the OMPL
        # scene contained the robot and NOTHING else: no table, no objects, and no
        # payload after a grasp. "Collision-free motion planning" therefore meant
        # only *self*-collision-free, and the arm could (and did) plan straight
        # through the table. We now publish the surfaces (static, from the same scene
        # config the planner uses) and the perceived objects as CollisionObjects, and
        # attach the carried object to the hand while it is held.
        self._scene_objects = {}      # object_id -> CollisionObject
        self._attached_id = None
        self._surfaces_loaded = False
        self._surface_ids = []
        self._scene_cfg = os.environ.get("TAMP_SCENE_CONFIG", "")
        self.create_subscription(WorldState, "/world_state", self._on_world, 10,
                                 callback_group=self._io_cb)
        self._load_surfaces()

        self.get_logger().info("MoveItPy ready; planning group 'panda_arm' (pipeline=ompl)")
        self.create_subscription(TaskAction, "/task_plan", self._on_action, 10,
                                 callback_group=self._action_cb)
        self.n = 0

    # ---------------- planning scene ----------------
    @staticmethod
    def _box(obj_id, xyz, dims, frame="panda_link0"):
        co = CollisionObject()
        co.header.frame_id = frame
        co.id = obj_id
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(d) for d in dims]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = [float(v) for v in xyz]
        pose.orientation.w = 1.0
        co.primitives = [prim]
        co.primitive_poses = [pose]
        co.operation = CollisionObject.ADD
        return co

    def _apply(self, *objs):
        """Insert collision objects into MoveItPy's planning scene."""
        try:
            with self.robot.get_planning_scene_monitor().read_write() as scene:
                for o in objs:
                    if isinstance(o, AttachedCollisionObject):
                        scene.process_attached_collision_object(o)
                    else:
                        # NOTE: the MoveItPy binding is `apply_collision_object`;
                        # there is no `process_collision_object` on PlanningScene.
                        scene.apply_collision_object(o)
                scene.current_state.update()
            return True
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"   (planning scene update failed: {exc})")
            return False

    GRIPPER_LINKS = ("panda_hand", "panda_leftfinger", "panda_rightfinger")

    def _allow_touch(self, object_id, allow=True):
        """Let the gripper (and the support surfaces) TOUCH this object.

        You cannot grasp an object you are not allowed to collide with: the grasp pose
        puts the fingers *around* the object, and a placement rests it *on* a surface.
        Both are collisions. Without this, adding the objects to the planning scene makes
        OMPL correctly refuse the descent ("PLAN FAILED (contact)") and the arm retreats
        upward without ever picking anything. This is the standard MoveIt
        allowed-touch-objects mechanism, scoped to just the object being manipulated."""
        names = list(self.GRIPPER_LINKS) + list(self._surface_ids)
        try:
            with self.robot.get_planning_scene_monitor().read_write() as scene:
                acm = scene.allowed_collision_matrix
                for n in names:
                    acm.set_entry(object_id, n, allow)
                scene.current_state.update()
            return True
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"   (ACM update failed: {exc})")
            return False

    def _load_surfaces(self):
        """Static colliders (table/tray) from the same scene config the planner uses."""
        if self._surfaces_loaded or not self._scene_cfg or not os.path.exists(self._scene_cfg):
            return
        try:
            with open(self._scene_cfg) as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"scene_config unreadable: {exc}")
            return
        boxes = [self._box(s["id"], s["pose"][:3], s["dims"]) for s in cfg.get("surfaces", [])]
        if boxes and self._apply(*boxes):
            self._surfaces_loaded = True
            self._surface_ids = [b.id for b in boxes]
            self.get_logger().info(
                f"planning scene: added {len(boxes)} surface(s): {self._surface_ids}")

    def _on_world(self, msg):
        """Keep the perceived objects in the planning scene (skip the carried one —
        it is an AttachedCollisionObject and must not also exist as a world object)."""
        self._load_surfaces()
        boxes = []
        for o in msg.objects:
            if o.object_id == self._attached_id:
                continue
            p, d = o.pose.position, o.dimensions
            boxes.append(self._box(o.object_id, (p.x, p.y, p.z), (d.x, d.y, d.z)))
        if boxes:
            self._scene_objects = {b.id: b for b in boxes}
            self._apply(*boxes)

    def _attach(self, object_id):
        """Attach the carried object to the hand so post-grasp motions plan with the
        payload in the scene (previously the carried cup was invisible to OMPL)."""
        co = self._scene_objects.get(object_id)
        if co is None:
            return
        aco = AttachedCollisionObject()
        aco.link_name = "panda_hand"
        aco.object = co
        aco.object.operation = CollisionObject.ADD
        aco.touch_links = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
        if self._apply(aco):
            self._attached_id = object_id
            self.get_logger().info(f"   planning scene: attached '{object_id}' to panda_hand")

    def _detach(self, object_id):
        if self._attached_id != object_id:
            return
        aco = AttachedCollisionObject()
        aco.link_name = "panda_hand"
        aco.object.id = object_id
        aco.object.operation = CollisionObject.REMOVE
        self._apply(aco)
        self._attached_id = None
        self.get_logger().info(f"   planning scene: detached '{object_id}'")

    def _on_grasp_status(self, msg):
        self._grasp_status = msg.data      # "<ok|fail>:<obj>:<reason>"

    def _await_grasp_status(self, timeout=12.0):
        # Must exceed grasp_manager's worst-case latency: it shells out to
        # `gz topic -e` (4 s timeout) to read ground-truth model poses before it can
        # decide. A 3 s wait timed out *just before* the success arrived, so a grasp
        # that had actually attached (and that the drop-detector then confirmed was
        # held at 0 mm drift) was reported as a failure.
        """Wait for grasp_manager's verdict on the attach we just commanded.
        Returns (ok, reason). A timeout is a FAILURE, not a success — refusing to
        assume success is the whole point: a grasp that silently attached nothing was
        previously still reported as a completed pick-and-place."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._grasp_status is not None:
                parts = self._grasp_status.split(":", 2)
                ok = parts[0] == "ok"
                reason = parts[2] if len(parts) > 2 else parts[0]
                return ok, reason
            time.sleep(0.02)
        return False, "no_grasp_status(timeout)"

    def _on_action(self, msg):
        self.n += 1
        oid, act = msg.object_id, msg.action_type
        p = msg.target_pose.position
        self.get_logger().info(
            f"[step {self.n}] {act} {oid} -> target ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")
        if self.pickplace:
            ok, reason = self._pick_place_sequence(msg)
        else:
            ok = self._move_to(msg.target_pose, f"{act} {oid}")
            reason = "done" if ok else "plan_failed"
        self._publish_status(act, oid, ok, reason)

    def _publish_status(self, act, oid, ok, reason):
        """Report the step outcome so the planner can close the loop (A2)."""
        self.status_pub.publish(String(data=f"{'ok' if ok else 'fail'}:{act}:{oid}:{reason}"))
        self.get_logger().info(f"   /execution_status {'ok' if ok else 'FAIL'}:{act}:{oid}:{reason}")

    def _pick_place_sequence(self, msg):
        """Approach -> contact -> (grasp|release) -> retreat, with a TCP offset so
        panda_hand stops above the object (its frame sits ~0.10 m above the TCP).
        Returns (ok, reason) for the execution-status report."""
        tcp = 0.10
        contact = self._offset(msg.target_pose, tcp)               # hand above object/dest
        approach = self._offset(msg.target_pose, tcp + self.approach_z)
        retreat = self._offset(msg.target_pose, tcp + self.lift_z)
        is_pick = msg.action_type in ("PICK", "UNSTACK")

        # The object being manipulated must be touchable: a grasp puts the fingers
        # AROUND it and a place rests it ON a surface. Both are collisions, so without
        # this OMPL refuses the descent and the arm just retreats upward.
        self._allow_touch(msg.object_id, True)

        if is_pick:
            self._gripper(self.GRIPPER_OPEN, "open")     # open before reaching
        if not self._move_to(approach, "approach"):
            self._allow_touch(msg.object_id, False)
            self._move_to(retreat, "retreat")
            return False, "plan_failed"                  # couldn't even plan the approach
        if is_pick:
            # Centre ABOVE the object BEFORE descending: the straight-down contact move
            # inherits the approach's lateral error (up to the 20 mm arrival tolerance),
            # and a descent that lands >13 mm off-centre drags the cup sideways with the
            # near finger's contact field before the at-contact re-centring can help.
            for _ in range(3):
                xy = self._hand_xy_error(approach)
                if xy is None or xy <= 0.006:
                    break
                self.get_logger().info(
                    f"   re-centring approach: lateral err {xy*1000:.0f} mm > 6 mm")
                self._move_to(approach, "approach (re-centre)")
        reached = self._move_to(contact, "contact")
        if is_pick and not reached:
            # don't fire grasp if we didn't reach the object (else we'd attach
            # whatever happens to be near the gripper, e.g. an already-placed object)
            self.get_logger().warn(f"   skip grasp: did not reach {msg.object_id}")
            self._move_to(retreat, "retreat")
            return False, "not_reached"                  # -> planner falsifies reachable
        if is_pick:
            # CENTRE before closing. The weld attaches at the CURRENT relative pose,
            # so lateral arrival error becomes a permanently off-centre carry — the
            # instrumented fling had tcp_err=38 mm, which left the cup intersecting
            # finger geometry even fully open, and detach catapulted it. A re-plan of
            # the same contact pose from ~15 mm away consistently lands 6-11 mm, so
            # iterate up to twice while the LATERAL error exceeds 9 mm (free travel
            # to first contact is only 13 mm: fingers open ±38 mm, cup half-width 25).
            for _ in range(3):
                xy = self._hand_xy_error(contact)
                if xy is None or xy <= 0.006:
                    break
                self.get_logger().info(
                    f"   re-centring: lateral err {xy*1000:.0f} mm > 6 mm — re-planning contact")
                self._move_to(contact, "contact (re-centre)")
            self._gripper(self.GRIPPER_CLOSE, "close")   # fingers grip the object
            p = msg.target_pose.position                 # the object's grasp position
            self._grasp_status = None                    # arm the verdict
            self._grasp(f"grasp:{msg.object_id}:{p.x:.4f},{p.y:.4f},{p.z:.4f}")
            ok, reason = self._await_grasp_status()
            if not ok:
                # THE failure the kinematic attach can actually detect. Reporting it
                # (instead of assuming success) is what stops the pipeline declaring a
                # pick-and-place that moved nothing "complete".
                self.get_logger().error(f"   GRASP FAILED: {reason}")
                # ALWAYS release on a failure path. The attach may in fact have happened
                # (e.g. we merely timed out waiting for the verdict), and without this the
                # physics joint stays welded: the fingers open but the object hangs off the
                # gripper in mid-air, and the next PICK would attach a second one.
                # Invariant: every failure path leaves the gripper EMPTY.
                self._grasp(f"release:{msg.object_id}")
                self._gripper(self.GRIPPER_OPEN, "open")
                self._detach(msg.object_id)
                self._move_to(retreat, "retreat")
                self._allow_touch(msg.object_id, False)
                return False, "dropped"                  # -> planner falsifies graspable
            # CAGE the object once the weld holds it: re-command the fingers ~2 mm
            # PAST the stall so they keep a light, effort-capped squeeze for the whole
            # carry. This deliberately REVERSES the earlier "back off 1.5 mm" policy.
            # History: backing off was introduced when detach flung cups metres — but
            # that fling was later root-caused to margin-geometry (fixed: 40 mm cups,
            # 0.042 open, re-centring), and the release path now opens fully and
            # settles 1 s before detaching, so the discharge path is closed. What the
            # back-off actually did in the measured batch was let the VIOLABLE weld
            # drift unresisted: the cup slid 57 mm / 229 mm off its welded pose
            # mid-carry and the drop detector aborted the run. With real friction on
            # the cups (bullet tags, mu 3.0) a light finger cage adds a fricative grip
            # that resists exactly that drift; the 40 N effort cap keeps it gentle.
            # Calibration history, both extremes measured: backing the fingers OFF
            # 1.5 mm let the violable weld drift unresisted (2/6 trials dropped the cup
            # 57/229 mm mid-carry); squeezing 2 mm DEEPER re-created the constraint-
            # conflict discharge (cup flung 2 m at release). Holding EXACTLY at the
            # stall keeps the fingers as static walls at contact distance: no added
            # force to store, but drift must push past them.
            held = self._finger
            if held is not None:
                self._gripper(held, "hold at stall (cage without squeeze)")
            # The payload is now part of the robot: every subsequent motion must plan
            # with it attached, or the carried object is invisible to OMPL.
            self._attach(msg.object_id)
        else:
            if not reached:
                # We are still HOLDING the object. Put it down before retreating, or it
                # stays welded to the gripper and the replan's next PICK attaches a
                # second object at the same TCP.
                self._grasp(f"release:{msg.object_id}")
                self._gripper(self.GRIPPER_OPEN, "open")
                self._detach(msg.object_id)
                self._move_to(retreat, "retreat")
                return False, "not_reached"              # -> planner falsifies placeable/stackable
            # Open the fingers BEFORE detaching: releasing first left the object as a
            # free physics body inside fingers commanded to a 44 mm gap around a 50 mm
            # cup for >1 s, which squirts it sideways.
            self._gripper(self.GRIPPER_OPEN, "open")     # let go
            # SETTLE before detaching. DetachJoint re-enables parent<->child collision
            # THE INSTANT the weld is removed (gz-physics #640): if the object still
            # intersects any hand/finger geometry at that instant — which the violable
            # weld's mid-carry drift makes common — the deep-penetration impulse flings
            # it metres (instrumented capture: 2 m within 1 s of detach, arm still).
            # With the fingers fully open and the object hanging on the weld directly
            # above its support, a short settle lets the residual drift relax so
            # collisions re-enable on a non-penetrating state.
            time.sleep(1.0)
            self._grasp(f"release:{msg.object_id}")
            ok, reason = self._await_grasp_status()
            self._detach(msg.object_id)                  # no longer part of the robot
            if not ok:
                self.get_logger().error(f"   RELEASE FAILED: {reason}")
                self._move_to(retreat, "retreat")
                return False, "dropped"
        time.sleep(0.3)
        retreated = self._move_to(retreat, "retreat")
        if not retreated:
            # Cartesian retreat can be unplannable from an awkward end pose (measured:
            # 1 in 6 trials died as "PLAN FAILED (retreat)"). A joint-space move to the
            # SRDF "ready" configuration is always well-defined, so fall back to it.
            self.get_logger().warn("   Cartesian retreat failed — joint-space 'ready' fallback")
            retreated = self._move_ready()
        self._allow_touch(msg.object_id, False)          # restore normal collision checking
        if not retreated:
            # The pick/place itself SUCCEEDED (grasp verified / object released), so
            # this must stay ok — falsifying a bridge predicate for a retreat problem
            # would trigger a wrong replan. But silently returning "done" here hid a
            # real failure ("PLAN FAILED (retreat)" was followed by "ok:PLACE:done"),
            # so report it honestly in the reason and leave a loud log line.
            self.get_logger().warn(
                f"   step succeeded but RETREAT failed — arm may still be at the "
                f"{msg.action_type} pose; next step's approach will re-plan from here")
            return True, "done_retreat_failed"
        return True, "done"

    def _on_joints(self, msg):
        for name, p in zip(msg.name, msg.position):
            if name == "panda_finger_joint1":
                self._finger = float(p)
                return

    def _gripper(self, pos: float, label: str, timeout: float = 15.0):
        """Command the fingers and WAIT until they actually get there (or stall on the
        object). The old code slept a fixed 1.2 s of WALL clock against a 1 s SIM
        trajectory: whenever Gazebo's real-time factor dropped, the fingers were still
        moving when we checked, and the grasp was rejected as `fingers_not_closed`
        (observed at 0.0389 — barely off the 0.040 open position).

        A stall short of the target is not a failure when CLOSING — that is contact with
        the object, which is exactly what we want."""

        def send():
            # Command BOTH fingers to the same target via the FollowJointTrajectory
            # ACTION. joint2 used to be a URDF mimic of joint1, but Gazebo/DART silently
            # drops mimic constraints, so the right finger never actuated — the left
            # finger closed alone and stalled against the frozen right one. The hand
            # controller now owns both joints, so we send both for a symmetric close.
            # The ACTION (vs the best-effort /joint_trajectory topic) is what makes the
            # 2nd/3rd close reliable: the topic dropped later goals (finger stuck at
            # 0.0377); the action's accept/execute handshake does not.
            g = FollowJointTrajectory.Goal()
            jt = JointTrajectory()
            jt.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
            p = JointTrajectoryPoint()
            p.positions = [float(pos), float(pos)]
            p.time_from_start.sec = 1
            jt.points = [p]
            g.trajectory = jt
            fut = self.gripper_ac.send_goal_async(g)

            # DIAGNOSTIC: log whether the controller ACCEPTS the goal and what result
            # it returns. The 2nd grasp's close has zero effect while the 1st works —
            # this reveals if the goal is rejected, aborted, or silently no-op'd.
            def _on_goal(f):
                try:
                    gh = f.result()
                except Exception as exc:
                    self.get_logger().warn(f"   gripper {label}: send failed: {exc}")
                    return
                if gh is None or not gh.accepted:
                    self.get_logger().warn(f"   gripper {label}: goal REJECTED by controller")
                    return
                def _on_res(rf):
                    # Only surface FAILED results — a healthy run otherwise floods the
                    # log with one success line per finger goal.
                    try:
                        r = rf.result()
                        if r.result.error_code != 0:
                            self.get_logger().warn(
                                f"   gripper {label}: action code={r.result.error_code} "
                                f"'{r.result.error_string}'")
                    except Exception as exc:
                        self.get_logger().warn(f"   gripper {label}: result err: {exc}")
                gh.get_result_async().add_done_callback(_on_res)
            fut.add_done_callback(_on_goal)
            return fut

        if not self.gripper_ac.wait_for_server(timeout_sec=3.0):
            self.get_logger().error(f"   gripper {label}: action server unavailable")
            return self._finger
        send()
        # Log the finger position AT COMMAND TIME and the sim-clock reading. The
        # batch-context limit-pin (fingers at 0.04001, dead to all commands) needs
        # this to separate "controller drove them into the limit" from "they were
        # already pinned before we commanded anything".
        now = self.get_clock().now().nanoseconds * 1e-9
        self.get_logger().info(
            f"   gripper {label} ({pos:.3f}) [action]  fingers_now={self._finger} "
            f"sim_t={now:.2f}")

        closing = pos < (self.GRIPPER_OPEN - 1e-6)
        start = self._finger
        deadline = time.time() + timeout
        last, stable = None, 0
        # NO re-issue retry here — deliberately. An earlier version re-sent the command
        # every 1.5 s while the fingers had not yet travelled 3 mm (a workaround for the
        # /joint_trajectory TOPIC silently dropping goals). Once the gripper moved to the
        # ACTION interface that workaround became actively harmful: bullet drives the
        # fingers slower than 3 mm/1.5 s, so the retry kept firing and each new goal
        # PREEMPTED the in-flight trajectory, restarting the motion. The fingers never got
        # past ~0.4 mm and the grasp failed as fingers_not_closed:0.0376. The action has
        # an accept/execute handshake, so a goal cannot be silently dropped — just wait.
        while time.time() < deadline:
            f = self._finger
            if f is not None:
                if abs(f - pos) <= 0.002:                    # reached the command
                    self.get_logger().info(f"   fingers at {f:.4f}")
                    return f
                if last is not None and abs(f - last) < 2e-4:
                    stable += 1
                    # A stall only means CONTACT if the fingers actually MOVED first.
                    # Otherwise we are just seeing the interval before the controller
                    # has begun executing the 1 s trajectory — declaring that "contact"
                    # returned instantly with the fingers still wide open (0.0400), and
                    # the grasp was then rejected as fingers_not_closed.
                    moved = start is not None and abs(f - start) >= 0.003
                    if stable >= 8 and closing and moved:
                        self.get_logger().info(
                            f"   fingers stalled at {f:.4f} after moving "
                            f"{abs(f - start)*1000:.0f}mm (contact)")
                        return f
                else:
                    stable = 0
                last = f
            time.sleep(0.05)
        self.get_logger().warn(f"   gripper {label}: timed out at {self._finger} "
                               f"(started {start})")
        return self._finger

    def _grasp(self, cmd):
        self.grasp_pub.publish(String(data=cmd))
        self.get_logger().info(f"   /grasp_cmd '{cmd}'")
        time.sleep(0.4)

    @staticmethod
    def _offset(pose, dz):
        from copy import deepcopy
        p = deepcopy(pose)
        p.position.z = float(p.position.z) + dz
        return p

    def _move_to(self, pose, label):
        """Returns True only if the arm PLANNED **and actually arrived**.

        Previously this returned True as soon as OMPL found a path — so "reached" only
        ever meant "planned", `NOT_REACHED` never meant what feedback.py said it did,
        and no execution failure could ever be detected."""
        goal = PoseStamped()
        goal.header.frame_id = "panda_link0"
        goal.pose = pose
        if self.start_from_current:
            self.arm.set_start_state_to_current_state()
        else:
            self.arm.set_start_state(configuration_name="ready")
        self.arm.set_goal_state(pose_stamped_msg=goal, pose_link="panda_hand")
        self.get_logger().info(
            f"   {label}: plan to ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})")
        plan = self.arm.plan(single_plan_parameters=self.plan_params)
        if not plan:
            self.get_logger().error(f"   PLAN FAILED ({label})")
            return False
        traj = plan.trajectory
        self._publish_display(traj)
        if self.execute:
            reached, _err = self._execute_on_controller(traj, goal_pose=pose)
            return reached                 # <-- real arrival, not "a plan existed"
        elif self.dwell > 0:
            time.sleep(self.dwell)
        return True

    def _publish_display(self, traj):
        """Publish a DisplayTrajectory so RViz's MotionPlanning display animates it."""
        try:
            msg = DisplayTrajectory()
            msg.trajectory.append(traj.get_robot_trajectory_msg())
            self.display_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warn(f"   (display publish skipped: {exc})")

    def _execute_on_controller(self, traj, goal_pose=None):
        """Send the planned trajectory to the joint_trajectory_controller and WAIT
        UNTIL THE ARM ACTUALLY ARRIVES (not just until a wall-clock timer expires).

        Two bugs this fixes:
        * `time.sleep(secs)` timed a **sim-clock** trajectory with a **wall clock**.
          Every Gazebo launch runs `use_sim_time:=True` and RTF<1 is normal, so the
          executor returned EARLY and the next `_move_to` published a new trajectory
          (stamp=0) that *replaced the in-flight one*, causing a position discontinuity.
        * Arrival was never checked, so "reached" only ever meant "OMPL found a path".

        Returns (reached, position_error_m). Polls TF for the real hand pose.
        """
        jt = traj.get_robot_trajectory_msg().joint_trajectory
        if not jt.points:
            self.get_logger().warn("   (empty trajectory, nothing to execute)")
            return False, float("inf")
        # stamp 0 -> start now; controller follows time_from_start of each point
        jt.header.stamp.sec = 0
        jt.header.stamp.nanosec = 0
        dur = jt.points[-1].time_from_start
        secs = dur.sec + dur.nanosec * 1e-9
        # Send via the ACTION, not the topic (see the arm_ac comment): the best-effort
        # topic could drop/truncate the trajectory, leaving the arm short of the goal.
        if not self.arm_ac.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("   arm action server unavailable")
            return False, float("inf")
        g = FollowJointTrajectory.Goal()
        g.trajectory = jt
        self.arm_ac.send_goal_async(g)
        self.get_logger().info(f"   EXECUTING on controller (traj {secs:.1f}s sim) [action]")

        if goal_pose is None:            # nothing to verify against
            time.sleep(secs + 0.5)
            return True, 0.0

        # Poll the ACTUAL hand pose until it converges on the goal, or we give up.
        deadline = time.time() + max(secs * self.rtf_slack, 2.0) + self.settle_timeout
        best = float("inf")
        while time.time() < deadline:
            err = self._hand_position_error(goal_pose)
            if err is not None:
                best = min(best, err)
                if err <= self.reach_tol:
                    self.get_logger().info(f"   arrived (err {err*1000:.0f} mm)")
                    return True, err
            time.sleep(0.05)
        self.get_logger().warn(
            f"   DID NOT ARRIVE: best error {best*1000:.0f} mm > tol {self.reach_tol*1000:.0f} mm")
        return False, best

    def _move_ready(self):
        """Joint-space move to the SRDF 'ready' configuration — the always-plannable
        retreat when a Cartesian retreat pose has no OMPL solution."""
        try:
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(configuration_name="ready")
            plan = self.arm.plan(single_plan_parameters=self.plan_params)
            if not plan:
                self.get_logger().warn("   ready-retreat: plan failed")
                return False
            ok, _err = self._execute_on_controller(plan.trajectory, None)
            return ok
        except Exception as exc:
            self.get_logger().warn(f"   ready-retreat error: {exc}")
            return False

    def _hand_position_error(self, goal_pose):
        """Euclidean error between the real panda_hand origin and the goal (m)."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "panda_link0", "panda_hand", rclpy.time.Time())
        except Exception:
            return None
        t = tf.transform.translation
        p = goal_pose.position
        return math.sqrt((t.x - p.x) ** 2 + (t.y - p.y) ** 2 + (t.z - p.z) ** 2)

    def _hand_xy_error(self, goal_pose):
        """LATERAL (xy) error between the real panda_hand origin and the goal (m).
        Used to centre the gripper over the object before closing: vertical error
        barely matters for a top grasp, but lateral error becomes the weld's
        permanent off-centre offset."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "panda_link0", "panda_hand", rclpy.time.Time())
        except Exception:
            return None
        t = tf.transform.translation
        p = goal_pose.position
        return math.hypot(t.x - p.x, t.y - p.y)


def main():
    rclpy.init(args=sys.argv)
    # MoveItPy creates the "moveit_py" node, which (via the launch __node remap)
    # receives the MoveIt config parameters and loads the OMPL pipeline.
    robot = MoveItPy(node_name="moveit_py")
    node = MoveItExecutor(robot)
    # MultiThreadedExecutor is REQUIRED: _on_action blocks for a whole pick-and-place
    # while polling for the grasp verdict, so /grasp_status and /world_state must be
    # able to run concurrently (they are in a Reentrant group). With the default
    # single-threaded spin they were serialized behind _on_action and every grasp
    # timed out even when it had actually succeeded.
    from rclpy.executors import MultiThreadedExecutor
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
