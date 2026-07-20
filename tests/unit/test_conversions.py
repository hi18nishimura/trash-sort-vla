"""M4 bridge tests: ROS↔core conversions (duck-typed fakes, no rclpy needed).

Conversions live in the ROS-free ``tamp.ros_conversions`` module so they are
importable and testable without ROS installed.
"""
from dataclasses import dataclass, field
from typing import List

from tamp import ros_conversions as C
from tamp.types import Pose, TaskAction


# --- minimal duck-typed ROS message fakes ---
@dataclass
class _V3: x: float = 0.0; y: float = 0.0; z: float = 0.0
@dataclass
class _Quat: w: float = 1.0; x: float = 0.0; y: float = 0.0; z: float = 0.0
@dataclass
class _PoseMsg:
    position: _V3 = field(default_factory=_V3)
    orientation: _Quat = field(default_factory=_Quat)
@dataclass
class _ObjMsg:
    object_id: str = "A"; class_name: str = "block"
    pose: _PoseMsg = field(default_factory=_PoseMsg)
    dimensions: _V3 = field(default_factory=lambda: _V3(0.05, 0.05, 0.05))
    confidence: float = 0.9
@dataclass
class _WorldMsg:
    objects: List[_ObjMsg] = field(default_factory=lambda: [_ObjMsg()])
@dataclass
class _TaskMsg:
    action_type: str = ""; object_id: str = ""
    target_pose: _PoseMsg = field(default_factory=_PoseMsg)
    preconditions: list = field(default_factory=list)
    effects: list = field(default_factory=list)


def test_world_state_roundtrip_quaternion_order():
    m = _WorldMsg(objects=[_ObjMsg(pose=_PoseMsg(position=_V3(1, 2, 3),
                                                 orientation=_Quat(0.0, 1.0, 0.0, 0.0)))])
    w = C.world_state_from_ros(m)
    assert w.objects[0].pose.position == (1, 2, 3)
    # (w,x,y,z) ordering preserved
    assert w.objects[0].pose.orientation == (0.0, 1.0, 0.0, 0.0)
    assert w.objects[0].dimensions == (0.05, 0.05, 0.05)


def test_task_action_to_ros():
    a = TaskAction("PICK", "A", target_pose=Pose((0.5, 0, 0.4), (1, 0, 0, 0)),
                   preconditions=["(clear A)"], effects=["(holding g A)"])
    msg = C.task_action_to_ros(a, _TaskMsg, _PoseMsg, _V3, _Quat)
    assert msg.action_type == "PICK" and msg.object_id == "A"
    assert msg.target_pose.position.x == 0.5
    assert msg.preconditions == ["(clear A)"]


def test_surfaces_from_config():
    cfg = {"surfaces": [{"id": "table", "pose": [0.5, 0, 0.4, 1, 0, 0, 0],
                         "dims": [0.8, 0.8, 0.02], "class": "table"}]}
    surfs = C.surfaces_from_config(cfg)
    assert surfs[0].surface_id == "table"
    assert surfs[0].dimensions == (0.8, 0.8, 0.02)
