#!/usr/bin/env python3
"""Real perception: RGB-D camera + YOLOv8 (COCO) -> /world_state.

Replaces mock_perception. For each YOLO detection it samples the depth at the box
centre, back-projects to 3D using the camera intrinsics (camera_info), transforms
from the camera optical frame to the robot base (panda_link0) using the known,
fixed camera mount, and publishes an ObjectState. Mirrors the reference
object_state_manager but with YOLOv8-COCO detection instead of YOLOv8-OBB.

Runs under the venv python (ultralytics) with ROS on PYTHONPATH.
"""
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion

from pytamp_moveit_bridge.msg import ObjectState, WorldState

# Camera mount: pose of the camera LINK in panda_link0 frame (x,y,z, roll,pitch,yaw).
# Default = the perception/tabletop world's top-down camera at (0.5,0,1.0) pitch 90deg.
DEFAULT_CAM_XYZ = (0.5, 0.0, 1.0)
DEFAULT_CAM_RPY = (0.0, 1.5708, 0.0)
# Optical frame is rotated wrt the link: optical +Z = link +X (forward),
# optical +X = link -Y, optical +Y = link -Z (REP-103 camera convention).
_R_LINK_OPT = np.array([[0, 0, 1],
                        [-1, 0, 0],
                        [0, -1, 0]], dtype=float)

CONF_THRES = 0.25


def rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


class YoloPerception(Node):
    def __init__(self):
        super().__init__("yolo_perception")
        self.declare_parameter("model", "yolov8n.pt")
        self.declare_parameter("cam_xyz", list(DEFAULT_CAM_XYZ))
        self.declare_parameter("cam_rpy", list(DEFAULT_CAM_RPY))
        self.declare_parameter("classes", [])  # optional COCO class allowlist

        from ultralytics import YOLO
        import torch
        self.model = YOLO(self.get_parameter("model").value)
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Warm up on the GPU so the FIRST real detection isn't a ~0.8 s cold start.
        self.model(np.zeros((480, 640, 3), np.uint8), device=dev, verbose=False)
        self.get_logger().info(f"YOLO warmed up on {dev}")
        self.allow = set(self.get_parameter("classes").value or [])

        xyz = self.get_parameter("cam_xyz").value
        rpy = self.get_parameter("cam_rpy").value
        T = np.eye(4)
        T[:3, :3] = rpy_to_R(*rpy) @ _R_LINK_OPT          # base <- optical rotation
        T[:3, 3] = xyz
        self.T_base_opt = T

        self.K = None
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, 10)
        img = message_filters.Subscriber(self, Image, "/camera/image")
        depth = message_filters.Subscriber(self, Image, "/camera/depth_image")
        self.ts = message_filters.ApproximateTimeSynchronizer([img, depth], 10, 0.3)
        self.ts.registerCallback(self._on_rgbd)

        self.pub = self.create_publisher(WorldState, "/world_state", 10)
        self.get_logger().info("yolo_perception ready (YOLOv8-COCO -> /world_state)")

    def _on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f"camera intrinsics: fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f}")

    def _on_rgbd(self, img_msg, depth_msg):
        if self.K is None:
            return
        rgb = np.frombuffer(bytes(img_msg.data), np.uint8).reshape(img_msg.height, img_msg.width, 3)
        if img_msg.encoding == "bgr8":
            rgb = rgb[:, :, ::-1]
        depth = self._depth_to_m(depth_msg)

        import time
        _t = time.time()
        res = self.model(np.ascontiguousarray(rgb), verbose=False)[0]
        self.get_logger().info(f"YOLO: {len(res.boxes)} detection(s) in {(time.time()-_t)*1000:.0f} ms",
                               throttle_duration_sec=2.0)
        world = WorldState()
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        counts = {}
        for b in res.boxes:
            conf = float(b.conf)
            if conf < CONF_THRES:
                continue
            cls = self.model.names[int(b.cls)]
            if self.allow and cls not in self.allow:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
            d = self._sample_depth(depth, u, v)
            if d is None or d <= 0:
                continue
            # back-project to optical frame, then to base frame
            p_opt = np.array([(u - cx) / fx * d, (v - cy) / fy * d, d, 1.0])
            p_base = self.T_base_opt @ p_opt
            # rough metric size from box + depth
            w_m = (x2 - x1) / fx * d
            h_m = (y2 - y1) / fy * d
            counts[cls] = counts.get(cls, 0) + 1
            oid = f"{cls}_{counts[cls]}"
            world.objects.append(self._object(oid, cls, p_base[:3], (w_m, h_m, max(w_m, h_m)), conf))
        if world.objects:
            self.pub.publish(world)
            self.get_logger().info(
                "published /world_state: " + ", ".join(f"{o.object_id}@({o.pose.position.x:.2f},"
                f"{o.pose.position.y:.2f},{o.pose.position.z:.2f})" for o in world.objects))

    @staticmethod
    def _object(oid, cls, xyz, dims, conf):
        o = ObjectState()
        o.object_id, o.class_name, o.confidence = oid, cls, float(conf)
        o.pose = Pose(position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
                      orientation=Quaternion(w=1.0))
        o.dimensions = Vector3(x=float(dims[0]), y=float(dims[1]), z=float(dims[2]))
        return o

    def _depth_to_m(self, msg):
        if msg.encoding in ("32FC1", "32FC"):
            return np.frombuffer(bytes(msg.data), np.float32).reshape(msg.height, msg.width)
        if msg.encoding == "16UC1":
            return np.frombuffer(bytes(msg.data), np.uint16).reshape(msg.height, msg.width) / 1000.0
        raise ValueError(f"depth encoding {msg.encoding}")

    @staticmethod
    def _sample_depth(depth, u, v, win=3):
        h, w = depth.shape
        u = min(max(u, win), w - win - 1); v = min(max(v, win), h - win - 1)
        patch = depth[v - win:v + win + 1, u - win:u + win + 1]
        vals = patch[np.isfinite(patch) & (patch > 0)]
        return float(np.median(vals)) if vals.size else None


def main():
    rclpy.init(args=sys.argv)
    node = YoloPerception()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
