#!/usr/bin/env bash
# Reliability harness for the Gazebo pick-and-place (the `gather` goal).
#
# Runs ./run.sh pnp N times HEADLESS and, for each trial, records:
#   * whether the planner reported PLAN COMPLETE, and
#   * GROUND TRUTH from Gazebo: are BOTH cups actually resting on the tray?
#
# The second check is the one that matters — a run can report success while the
# cups never moved (that exact bug existed), so the log alone is not evidence.
#
#   bash scripts/reliability_pnp.sh [N]     # default 6
#
# Writes per-trial logs to /tmp/tamp_logs/rel_<i>.log and prints a summary table.
# NOTE: no `set -u` — ROS's /opt/ros/jazzy/setup.bash references unbound vars
# (AMENT_TRACE_SETUP_FILES) and aborts under it, which killed every trial instantly.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N="${1:-6}"
TIMEOUT="${TIMEOUT:-200}"          # seconds per trial
LOGDIR=/tmp/tamp_logs; mkdir -p "$LOGDIR"

# Tray geometry (worlds/tabletop_pnp_nocam.sdf): centre (0.55,0), top z=0.420,
# half-extent 0.17. Cup half-height 0.035 -> resting ON the tray means z=0.455.
TRAY_X=0.55; TRAY_Y=0.0; TRAY_HALF=0.17; REST_Z=0.455; Z_TOL=0.02

kill_sim() {
  # SIGINT the launch FIRST and give it time to shut its children down cleanly.
  # kill -9 on `ros2 launch` cannot be trapped, so it orphaned Gazebo mid-step
  # (looked like a crash), froze /clock (use_sim_time nodes appeared hung), and
  # leaked half-dead stacks into the next trial. Only stragglers that survive the
  # grace period get force-killed.
  pkill -INT -f "ros2 launch pytamp_moveit_bridge" 2>/dev/null
  # NOTE: loop variable must NOT be `i` — kill_sim runs inside the trial loop and a
  # shared `i` clobbered the trial counter (rows all printed "1" and every trial's
  # log overwrote rel_1.log).
  for _w in $(seq 1 10); do
    pgrep -f "ros2 launch pytamp_moveit_bridge" >/dev/null 2>&1 || break
    sleep 1
  done
  for p in $(pgrep -f "gz sim|gz-sim|ruby|ros2 launch|moveit_executor|grasp_manager|tamp_planner|parameter_bridge|robot_state_pub|spawner|ros_gz|mock_perception" 2>/dev/null); do
    [ "$p" != "$$" ] && kill -9 "$p" 2>/dev/null
  done
  sleep 2
}

cup_pose() {   # $1 = model name -> "x y z" (empty if not found)
  gz topic -e -t /world/tabletop/pose/info -n 1 2>/dev/null \
    | awk -v want="\"$1\"" '
        $1=="name:" && $2==want {n=1}
        n && /position/ {p=1}
        n && p && /^ *x:/ {x=$2}
        n && p && /^ *y:/ {y=$2}
        n && p && /^ *z:/ {print x, y, $2; exit}
        n && p && /^ *w:/ {n=0; p=0}'
}

on_tray() {    # $1 x  $2 y  $3 z  -> 0 if resting on the tray
  [ -z "${3:-}" ] && return 1
  awk -v x="$1" -v y="$2" -v z="$3" -v tx="$TRAY_X" -v ty="$TRAY_Y" \
      -v h="$TRAY_HALF" -v rz="$REST_Z" -v tol="$Z_TOL" \
      'BEGIN{ dx=(x-tx); if(dx<0)dx=-dx; dy=(y-ty); if(dy<0)dy=-dy;
              dz=(z-rz); if(dz<0)dz=-dz;
              exit !(dx<h && dy<h && dz<tol) }'
}

pass=0; declare -a ROWS=()
echo "=== pick-and-place reliability: $N headless trials (goal: BOTH cups on the tray) ==="

for i in $(seq 1 "$N"); do
  kill_sim
  LOG="$LOGDIR/rel_$i.log"; : > "$LOG"
  ( cd "$REPO" && source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash \
    && export TAMP_REPO="$REPO" TAMP_PYTHON="$REPO/.venv/bin/python3" \
    && export PYTHONPATH="${PYTHONPATH:-}:$REPO" \
    && ros2 launch pytamp_moveit_bridge tamp_pnp_mock.launch.py gui:=false ) > "$LOG" 2>&1 &

  t=0; verdict="TIMEOUT"
  while [ "$t" -lt "$TIMEOUT" ]; do
    if grep -q "PLAN COMPLETE" "$LOG" 2>/dev/null; then verdict="PLAN_COMPLETE"; break; fi
    if grep -q "TAMP FAILED"   "$LOG" 2>/dev/null; then verdict="TAMP_FAILED";   break; fi
    sleep 3; t=$((t+3))
  done
  sleep 4                       # let the last cup settle under gravity

  source /opt/ros/jazzy/setup.bash 2>/dev/null
  c1=$(cup_pose cup); c2=$(cup_pose cup2)
  g1="no"; g2="no"
  on_tray $c1 && g1="YES"
  on_tray $c2 && g2="YES"

  if [ "$verdict" = "PLAN_COMPLETE" ] && [ "$g1" = "YES" ] && [ "$g2" = "YES" ]; then
    res="PASS"; pass=$((pass+1))
  else
    res="FAIL"
  fi
  # first failure reason from the log, if any
  why=$(grep -oE "GRASP FAILED: [^ ]+|DID NOT ARRIVE|PLAN FAILED \([a-z]+\)|fingers_not_closed[^ ]*|no_object_within_reach[^ ]*|DROPPED" "$LOG" 2>/dev/null | head -1)
  ROWS+=("$(printf '%-3s %-6s %-14s cup=%-3s cup2=%-3s %s' "$i" "$res" "$verdict" "$g1" "$g2" "${why:-}")")
  echo "  ${ROWS[-1]}"
  kill_sim
done

echo ""
echo "=== SUMMARY ==="
printf '%s\n' "${ROWS[@]}"
echo ""
echo "  success: $pass / $N  ($(awk -v p="$pass" -v n="$N" 'BEGIN{printf "%.0f", 100*p/n}')%)"
echo "  PASS requires BOTH: planner reported PLAN COMPLETE *and* Gazebo ground truth"
echo "  shows both cups resting on the tray (z=0.455 +/- 0.02, inside the footprint)."
