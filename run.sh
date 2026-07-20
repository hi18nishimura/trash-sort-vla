#!/usr/bin/env bash
# One-command runner for the PDDL + PyTAMP TAMP demo.
#
#   ./run.sh                # offline plan (no ROS, instant)
#   ./run.sh ros            # full ROS pipeline, headless (prints PLAN OK)
#   ./run.sh rviz           # full ROS pipeline + RViz (watch the arm)  [needs a display]
#   ./run.sh rviz replan    # same, replan-recovery scene
#
# Everything (sourcing, the venv-vs-system-python split, auto-goal) is handled here.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-offline}"
SCENE="${2:-stack3}"
# allow `./run.sh ros replan` or `./run.sh replan` (scene as first arg)
case "$MODE" in stack3|replan) SCENE="$MODE"; MODE="offline";; esac

if [ "$MODE" = "offline" ]; then
  exec "$REPO/.venv/bin/python3" -c "import sys; sys.path.insert(0,'$REPO'); \
import runpy; sys.argv=['run_offline','--scene','$SCENE']; \
runpy.run_path('$REPO/scripts/run_offline.py', run_name='__main__')"
fi


# --- ROS modes ---
source /opt/ros/jazzy/setup.bash
source "$REPO/ros2_ws/install/setup.bash"
export TAMP_REPO="$REPO"
export TAMP_PYTHON="$REPO/.venv/bin/python3"
export TAMP_SCENE="$SCENE"
export PYTHONPATH="$PYTHONPATH:$REPO"

if [ "$MODE" = "gazebo" ]; then
  echo ">> Gazebo physics: scene=$SCENE  (arm executes trajectories; goal auto-sends ~22s; Ctrl-C to stop)"
  exec ros2 launch pytamp_moveit_bridge tamp_gazebo.launch.py scene:="$SCENE" gui:=true
fi

if [ "$MODE" = "pnp" ]; then
  # Watch real pick-and-place (cup -> tray) with the GUI, using known object poses
  # (no camera, so no GPU camera-render dependency). This is the verified manipulation demo.
  echo ">> Pick-and-place (mock perception, GUI): arm gathers both cups onto the tray (~40s)"
  exec ros2 launch pytamp_moveit_bridge tamp_pnp_mock.launch.py gui:=true
fi

if [ "$MODE" = "yolo" ]; then
  # Run with the GUI on your display; gz renders the camera via your desktop's
  # NVIDIA GL. (Do NOT force EGL/GLX vendor here — that's only a headless hack and
  # triggers an NV-GLX crash when a real X display is present.)
  echo ">> YOLO pick-and-place: camera->YOLOv8->TAMP->MoveIt->Gazebo (goal auto-sends ~30s; Ctrl-C to stop)"
  exec ros2 launch pytamp_moveit_bridge tamp_yolo_pnp.launch.py gui:=true
fi

RVIZ=false
[ "$MODE" = "rviz" ] && RVIZ=true

echo ">> launching scene=$SCENE rviz=$RVIZ  (goal auto-sends after ~15s; Ctrl-C to stop)"
exec ros2 launch pytamp_moveit_bridge tamp_demo.launch.py \
  scene:="$SCENE" rviz:="$RVIZ" auto_goal:=true
