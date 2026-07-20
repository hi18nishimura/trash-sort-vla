#!/usr/bin/env bash
# Standalone perception check: gz renders a camera over a few objects, bridge the
# image to ROS, run YOLOv8-COCO on one frame. Saves results/cam_frame.png + cam_yolo.png.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source "$REPO/ros2_ws/install/setup.bash"
WORLD="$REPO/ros2_ws/install/pytamp_moveit_bridge/share/pytamp_moveit_bridge/worlds/perception_test.sdf"
export GZ_SIM_RESOURCE_PATH="$(dirname "$(ros2 pkg prefix robot_description)/share")"

echo ">> starting gz (headless, GPU render)"
gz sim -s -r -v1 "$WORLD" > /tmp/tamp_ros/gz_perc.log 2>&1 &
GZ=$!
sleep 6

echo ">> bridging camera topics"
ros2 run ros_gz_bridge parameter_bridge \
  /camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo \
  /camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image \
  > /tmp/tamp_ros/bridge_perc.log 2>&1 &
BR=$!
sleep 4

echo ">> capturing + running YOLO"
"$REPO/.venv/bin/python3" "$REPO/scripts/capture_and_yolo.py" /camera/image

kill $GZ $BR 2>/dev/null
