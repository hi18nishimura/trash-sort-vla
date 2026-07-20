#!/usr/bin/env bash
# One real end-to-end pick-and-place, headless, with: a TIMESTAMPED streaming tracker
# of BOTH cups (to catch the exact moment of any fling), finger-joint sampling, grasp
# verdicts, and Gazebo ground truth at the end.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=/tmp/tamp_logs/pnp_trace.log; mkdir -p /tmp/tamp_logs

# --- graceful shutdown -------------------------------------------------------
# NEVER kill -9 `ros2 launch`: SIGKILL cannot be trapped, so launch dies without
# cleaning up its children — Gazebo gets orphaned mid-step (we mistook that for a
# crash), /clock freezes (use_sim_time nodes then look "hung"), and half-dead
# stacks leak into the next run. SIGINT lets launch shut everything down properly;
# only stragglers that survive the grace period get force-killed.
stop_stack() {
  pkill -INT -f "ros2 launch pytamp_moveit_bridge" 2>/dev/null
  for _w in $(seq 1 10); do
    pgrep -f "ros2 launch pytamp_moveit_bridge" >/dev/null 2>&1 || break
    sleep 1
  done
  for pat in "gz sim" "gz-sim" "ruby" "ros2 launch" "moveit_executor" \
             "grasp_manager" "tamp_planner" "spawner" "ros_gz" "mock_perception"; do
    for p in $(pgrep -f "$pat" 2>/dev/null); do
      [ "$p" != "$$" ] && kill -9 "$p" 2>/dev/null
    done
  done
  sleep 1
}

stop_stack   # start clean

source /opt/ros/jazzy/setup.bash; source "$REPO/ros2_ws/install/setup.bash"
export TAMP_REPO="$REPO" TAMP_PYTHON="$REPO/.venv/bin/python3"
( cd "$REPO" && ros2 launch pytamp_moveit_bridge tamp_pnp_mock.launch.py gui:=false ) > "$LOG" 2>&1 &

# --- timestamped streaming tracker for BOTH cups ------------------------------
# One long-lived `gz topic -e` STREAM (not a poll loop spawning a process per
# sample). dynamic_pose/info only carries non-static models, so the cups dominate.
# Timestamps let the fling moment be correlated against event lines in the launch
# log (grasp/release/detach/PLACE). Uses the gz CLI because the ros_gz bridge
# publishes TFMessages with EMPTY child_frame_id (model names lost).
( gz topic -e -t /world/tabletop/dynamic_pose/info 2>/dev/null \
  | stdbuf -oL awk '
      $1=="name:" {n=$2; gsub(/"/,"",n); inpos=0}
      /position {/ {inpos=1}
      /orientation {/ {inpos=0}
      inpos && (n=="cup" || n=="cup2") {
        if ($1=="x:") x=$2; if ($1=="y:") y=$2;
        if ($1=="z:") { printf "%s %s x=%s y=%s z=%s\n", strftime("%H:%M:%S"), n, x, y, $2; fflush() }
      }' ) > /tmp/tamp_logs/cup_track.log 2>&1 &
CUPTRK=$!

# sample BOTH fingers every 0.5s in the background for the whole run
( source /opt/ros/jazzy/setup.bash
  for k in $(seq 1 240); do
    ros2 topic echo /joint_states --once 2>/dev/null | sed '/^\t/d' | sed -n '/^---/q;p' \
     | python3 -c "
import sys,yaml
try:
    d=yaml.safe_load(sys.stdin.read()); n,p=list(d['name']),list(d['position'])
    f1=p[n.index('panda_finger_joint1')] if 'panda_finger_joint1' in n else None
    f2=p[n.index('panda_finger_joint2')] if 'panda_finger_joint2' in n else None
    if f1 is not None: print(f'FINGERS f1={f1:.4f} f2={f2:.4f}')
except Exception: pass"
    sleep 0.5
  done ) > /tmp/tamp_logs/fingers.log 2>&1 &
SAMP=$!

t=0; verdict=TIMEOUT
while [ "$t" -lt 220 ]; do
  grep -q "PLAN COMPLETE" "$LOG" 2>/dev/null && { verdict=PLAN_COMPLETE; break; }
  grep -q "TAMP FAILED"   "$LOG" 2>/dev/null && { verdict=TAMP_FAILED;   break; }
  sleep 3; t=$((t+3))
done
sleep 4

echo "=== VERDICT: $verdict (after ${t}s) ==="
echo ""
echo "=== grasp verdicts (grasp_manager) ==="
grep -oE "grasped '[^']+'[^—]*—[^)]*|/grasp_status FAIL[^ ]*|object_not_between_fingers[^ ]*|fingers_not_closed:[0-9.]+" "$LOG" | head
echo ""
echo "=== falsifications (now visible thanks to the logging fix) ==="
grep -oE "falsifying [^)]*\)|falsified=\[[^]]*\]" "$LOG" | head -5

echo ""
echo "=== Gazebo ground truth (cup positions) ==="
for m in cup cup2; do
  out=$(gz topic -e -t /world/tabletop/pose/info -n 1 2>/dev/null | awk -v want="\"$m\"" '
    $1=="name:" && $2==want {n=1}
    n && /position/ {p=1}
    n && p && /^ *x:/{x=$2} n && p && /^ *y:/{y=$2}
    n && p && /^ *z:/{printf "x=%.3f y=%.3f z=%.3f", x, y, $2; exit}')
  if [ -z "$out" ]; then
    echo "  $m: <PARSE RETURNED NOTHING — ground-truth check would wrongly say 'no'>"
  else
    echo "  $m: $out"
  fi
done
echo "  ON TRAY means: |x-0.55|<0.17, |y|<0.17, z=0.455+/-0.02  (tray top 0.420 + cup half 0.035)"
echo "  cups START at (0.45,+/-0.20,0.445) on the table"

kill "$SAMP" 2>/dev/null; kill "$CUPTRK" 2>/dev/null
stop_stack
