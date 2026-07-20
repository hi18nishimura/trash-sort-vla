"""Conversions between ROS 2 messages and the ROS-free ``tamp`` core types.

Kept import-light (no ``rclpy``) so the conversion logic is unit-testable off a
ROS machine. The ROS message classes are duck-typed: any object exposing the
same fields as ``pytamp_moveit_bridge/{WorldState,ObjectState,TaskAction}`` and
``geometry_msgs/Pose`` works.
"""
from __future__ import annotations

from tamp.types import ObjectState, Pose, Surface, TaskAction, WorldState


def pose_from_ros(p) -> Pose:
    return Pose(
        position=(p.position.x, p.position.y, p.position.z),
        orientation=(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z),
    )


def pose_to_ros(pose: Pose, PoseMsg, PointMsg, QuatMsg):
    m = PoseMsg()
    m.position = PointMsg(x=float(pose.position[0]), y=float(pose.position[1]), z=float(pose.position[2]))
    w, x, y, z = pose.orientation
    m.orientation = QuatMsg(w=float(w), x=float(x), y=float(y), z=float(z))
    return m


def object_state_from_ros(o) -> ObjectState:
    return ObjectState(
        object_id=o.object_id,
        class_name=o.class_name,
        pose=pose_from_ros(o.pose),
        dimensions=(o.dimensions.x, o.dimensions.y, o.dimensions.z),
        confidence=o.confidence,
    )


def world_state_from_ros(msg, surfaces=None) -> WorldState:
    """Convert a ``WorldState`` message to the core type. Surfaces are not in the
    message (the reference assumes them implicitly) so they are injected from the
    scene config."""
    return WorldState(
        objects=[object_state_from_ros(o) for o in msg.objects],
        surfaces=list(surfaces or []),
    )


def task_action_to_ros(action: TaskAction, TaskActionMsg, PoseMsg, PointMsg, QuatMsg):
    m = TaskActionMsg()
    m.action_type = action.action_type
    m.object_id = action.object_id
    if action.target_pose is not None:
        m.target_pose = pose_to_ros(action.target_pose, PoseMsg, PointMsg, QuatMsg)
    m.preconditions = list(action.preconditions)
    m.effects = list(action.effects)
    return m


def surfaces_from_config(cfg) -> list:
    """Build ``Surface`` list from a YAML/dict scene config:
        surfaces:
          - {id: table, pose: [x,y,z,qw,qx,qy,qz], dims: [dx,dy,dz], class: table}
    """
    out = []
    for s in cfg.get("surfaces", []):
        p = s["pose"]
        pose = Pose(position=tuple(p[:3]),
                    orientation=tuple(p[3:7]) if len(p) >= 7 else (1, 0, 0, 0))
        out.append(Surface(surface_id=s["id"], pose=pose,
                           dimensions=tuple(s["dims"]), class_name=s.get("class", "table")))
    return out
