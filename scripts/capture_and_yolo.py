#!/usr/bin/env python3
"""Subscribe one camera frame from the ROS image topic, save it, run YOLOv8-COCO,
print detections, and save an annotated image. Standalone perception check.

Run under the venv python (ultralytics) with ROS on PYTHONPATH:
    .venv/bin/python3 scripts/capture_and_yolo.py /camera/image
"""
import sys
import os

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def img_to_np(msg: Image) -> np.ndarray:
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = arr.reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1]
        return np.ascontiguousarray(img)
    raise ValueError(f"unsupported encoding {msg.encoding}")


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "/camera/image"
    rclpy.init()
    node = rclpy.create_node("capture_yolo")
    frames = []
    node.create_subscription(Image, topic, lambda m: frames.append(m), 10)
    node.get_logger().info(f"waiting for an image on {topic} ...")
    import time
    end = time.time() + 30
    while rclpy.ok() and not frames and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not frames:
        print("NO IMAGE RECEIVED"); return
    img = img_to_np(frames[-1])
    print(f"got frame {img.shape}, encoding={frames[-1].encoding}")

    os.makedirs(OUT_DIR, exist_ok=True)
    import cv2
    cv2.imwrite(os.path.join(OUT_DIR, "cam_frame.png"), img[:, :, ::-1])  # save as BGR

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # COCO-pretrained, auto-downloads
    res = model(img, verbose=False)[0]
    print("=== detections ===")
    for b in res.boxes:
        cls = model.names[int(b.cls)]
        conf = float(b.conf)
        xyxy = [round(float(v), 1) for v in b.xyxy[0]]
        print(f"  {cls:15s} conf={conf:.2f} box={xyxy}")
    annotated = res.plot()  # BGR
    cv2.imwrite(os.path.join(OUT_DIR, "cam_yolo.png"), annotated)
    print(f"saved results/cam_frame.png and results/cam_yolo.png")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
