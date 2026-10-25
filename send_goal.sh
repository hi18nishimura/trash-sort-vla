#!/bin/bash
# 使い方: ./send_goal.sh <ゴミの種類> <ゴミ箱>
# 例:     ./send_goal.sh paper bin_orange
ros2 topic pub --once /planning_goal std_msgs/msg/String \
  "data: '{\"type\":\"place_class\",\"class\":\"$1\",\"surface\":\"$2\"}'"
