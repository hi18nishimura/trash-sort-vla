#!/usr/bin/env bash
# End-to-end Phase-1 demo on ROS 2 Jazzy:
#   mock_perception -> /world_state -> tamp_planner (PDDL+PyTAMP) -> /task_plan
#                                                                      |
#                                          moveit_executor (MoveItPy / OMPL)
#
# Usage:  scripts/run_ros_full.sh [stack3|replan]
# Needs:  ROS 2 Jazzy + the built workspace + the .venv (geometry deps).
SCENE="${1:-stack3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=/tmp/tamp_ros; mkdir -p "$LOG"

source /opt/ros/jazzy/setup.bash
source "$REPO/ros2_ws/install/setup.bash"
export TAMP_REPO="$REPO"
export PYTHONPATH="$PYTHONPATH:$REPO"
VENV="$REPO/.venv/bin/python3"
NODES="$REPO/ros2_ws/install/pytamp_moveit_bridge/lib/pytamp_moveit_bridge"
CFG="$REPO/ros2_ws/src/pytamp_moveit_bridge/config/scene_${SCENE}.yaml"

GOAL_stack3='{"type":"stack","order":["A","B","C"],"base_surface":"table"}'
GOAL_replan='{"type":"on_any","object":"block","surfaces":["tray_near","tray_far"]}'
GOAL_VAR="GOAL_${SCENE}"; GOAL="${!GOAL_VAR}"

# 1) MoveIt + perception + logger + executor
ros2 launch pytamp_moveit_bridge tamp_moveit.launch.py scene:="$SCENE" > "$LOG/stack.log" 2>&1 &
echo "[run] waiting for MoveItPy..."
for i in $(seq 1 40); do grep -q "MoveItPy ready" "$LOG/stack.log" && break; sleep 1; done

# 2) TAMP planner under the venv interpreter (pykin/trimesh/numpy<2 deps)
"$VENV" "$NODES/tamp_planner_node.py" --ros-args -p scene_config:="$CFG" > "$LOG/planner.log" 2>&1 &
echo "[run] waiting for planner..."
for i in $(seq 1 30); do grep -q "TAMP planner ready" "$LOG/planner.log" && break; sleep 1; done
sleep 3

# 3) trigger
echo "[run] publishing goal: $GOAL"
ros2 topic pub --once /planning_goal std_msgs/msg/String "data: '$GOAL'" >/dev/null 2>&1
sleep 30

echo "===== TAMP PLANNER ====="; grep -E "planning for|SOLVED|FAILED|falsified" "$LOG/planner.log" | tail -3
echo "===== MoveIt OMPL EXECUTION ====="; grep -E "planning to|PLAN OK|PLAN FAILED" "$LOG/stack.log" | tail -10

# cleanup
kill %1 %2 2>/dev/null
