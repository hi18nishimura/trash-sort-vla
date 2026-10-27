#!/bin/bash
# Keep collecting until every train pair has TARGET episodes, restarting the
# whole simulator stack whenever collect_demos.py asks for it (exit code 2).
# usage: tools/collect_loop.sh 30 ~/projects/panda-tamp/data/raw
TARGET=${1:-30}
OUT=${2:-$HOME/projects/panda-tamp/data/raw}
shift 2   # any further arguments are passed to collect_demos.py
cd ~/projects/panda-tamp
source scripts/ros_env.sh
source ros2_ws/install/setup.bash
mkdir -p "$OUT/launch_logs"

cleanup() {
  pkill -f "ros2 launch" ; pkill -f collect_demos
  sleep 3
  # kill EVERYTHING the launch starts, hard, so nothing piles up across rounds
  for p in "gz sim" parameter_bridge robot_state_publisher moveit_executor \
           tamp_planner_node mock_perception grasp_manager spawner ros_gz_sim; do
    pkill -9 -f "$p"
  done
  sleep 3
}

health() {
  echo "  health: gz=$(pgrep -fc 'gz sim') bridge=$(pgrep -fc parameter_bridge)" \
       "rsp=$(pgrep -fc robot_state_publisher) mem_free=$(free -m | awk '/Mem/{print $7}')MB"
}

for round in $(seq 1 500); do
  echo "===== round $round  $(date +%H:%M:%S) ====="
  cleanup
  health
  ros2 launch pytamp_moveit_bridge trash_sort.launch.py gui:=false item:=none \
      > "$OUT/launch_logs/round_$round.log" 2>&1 &
  timeout -k 30 1500 python3 tools/collect_demos.py --target "$TARGET" --out "$OUT" \
      --max-attempts 25 --max-consec-fail 3 "$@"
  code=$?
  cleanup
  if [ $code -eq 0 ]; then
    echo "===== all targets reached  $(date +%H:%M:%S) ====="
    break
  fi
done
