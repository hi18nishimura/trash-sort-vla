#!/usr/bin/env bash
# Source this to set up the combined ROS 2 Jazzy + venv environment for running
# the TAMP nodes (rclpy from ROS, pykin/trimesh/numpy<2/unified-planning from venv).
#   source scripts/ros_env.sh
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source "$REPO/ros2_ws/install/setup.bash"
export PYTHONPATH="$PYTHONPATH:$REPO"
# Use the venv interpreter for nodes (it has the TAMP geometry deps).
export TAMP_PYTHON="$REPO/.venv/bin/python3"
echo "ROS_DISTRO=$ROS_DISTRO  TAMP_PYTHON=$TAMP_PYTHON  REPO=$REPO"
