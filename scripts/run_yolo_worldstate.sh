#!/usr/bin/env bash
# Verify yolo_perception 3D projection: gz renders the angled camera, bridge
# RGB+depth+info, run the YOLO node, and echo /world_state vs ground truth.
# Ground truth (perception_test.sdf): ball(0.5,-0.15,0.06) bottle(0.5,0.05,0.10) cup(0.5,0.22,0.05)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source "$REPO/ros2_ws/install/setup.bash"
export PYTHONPATH="$PYTHONPATH:$REPO"
WORLD="$REPO/ros2_ws/install/pytamp_moveit_bridge/share/pytamp_moveit_bridge/worlds/perception_test.sdf"

gz sim -s -r -v1 "$WORLD" > /tmp/tamp_ros/gz_ws.log 2>&1 &
GZ=$!; sleep 6
ros2 run ros_gz_bridge parameter_bridge \
  /camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image \
  /camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo \
  > /tmp/tamp_ros/bridge_ws.log 2>&1 &
BR=$!; sleep 4

# YOLO node: camera in WORLD frame at the angled pose; base_frame=world (no robot).
"$REPO/.venv/bin/python3" "$REPO/ros2_ws/install/pytamp_moveit_bridge/lib/pytamp_moveit_bridge/yolo_perception.py" \
  --ros-args -p base_frame:=world \
  -p cam_xyz:="[0.0, 0.0, 0.55]" -p cam_rpy:="[0.0, 0.6, 0.0]" \
  > /tmp/tamp_ros/yolo_node.log 2>&1 &
YN=$!; sleep 12

echo "===== /world_state from YOLO ====="
timeout 6 ros2 topic echo --once /world_state 2>/dev/null | grep -E "object_id|class_name|x:|y:|z:" | head -40
echo "===== node log ====="
grep -E "intrinsics|published" /tmp/tamp_ros/yolo_node.log | tail -5
kill $GZ $BR $YN 2>/dev/null
