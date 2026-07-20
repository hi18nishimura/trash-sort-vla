# How to run

One script — `./run.sh` — handles all the sourcing, the venv/system-Python split,
and auto-sends the goal. Run it from the repo root (`~/Desktop/PDDL`).

```bash
./run.sh                 # offline: just print the TAMP plan (no ROS, instant)
./run.sh replan          # offline, the replan-recovery scene
./run.sh ros             # full ROS pipeline, headless (prints PLAN OK)
./run.sh ros replan      # ... replan scene
./run.sh rviz            # full ROS pipeline + RViz — watch the arm move
./run.sh gazebo          # Gazebo physics — the Panda physically executes the plan
./run.sh gazebo replan   # ... replan scene
./run.sh pnp             # WATCH real pick&place: arm gathers cups onto the tray (Gazebo GUI)
./run.sh yolo            # FULL: camera -> YOLOv8(COCO) -> TAMP -> MoveIt -> pick&place in Gazebo
```

**All of these open a GUI on your display** (`gazebo`/`pnp`/`yolo` → Gazebo
windows; `rviz` → RViz). Run them in your own desktop
terminal so the windows appear. Headless runs are only for automated verification.

**`./run.sh pnp`** is the verified, watch-it-now pick-and-place: known object poses
(no camera), GUI on. The Panda picks the cup and places it on the tray under
Gazebo physics (kinematic grasp-attach). Use this to see manipulation working.

**`./run.sh yolo`** adds the real camera+YOLO front-end. It needs the gz camera
**sensor** to render offscreen via EGL on the GPU. On some hybrid-GPU setups gz's
sensor EGL picks the wrong driver (`libEGL warning: ... driver (null)`) and no
camera frames are produced — then YOLO gets nothing. If you hit that, the
manipulation half is identical to `./run.sh pnp`; for the camera, try forcing the
NVIDIA EGL vendor in your shell before launching:
```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
./run.sh yolo
```
(Perception itself is verified: YOLOv8-COCO detects the cup/ball and back-projects
to accurate 3D `/world_state` when the sensor renders.)

**`./run.sh yolo`** is the capstone: a Gazebo RGBD camera feeds YOLOv8-COCO, which
detects objects (a cup, a ball) and publishes their 3D poses to `/world_state`;
TAMP plans "pick a cup, place it on the tray"; MoveIt executes the
pick→lift→place→retreat; a grasp manager attaches the object to the gripper
(kinematic grasp) so it's carried and placed. Needs the GPU (camera render +
YOLO) and ~6 GB free RAM. Best run with the GUI on your display.

**Gazebo note:** the arm physically executes each planned trajectory under
physics (verified). Object grasping (blocks being lifted) is **not** simulated
yet — the arm moves to each pick/place pose, but objects aren't attached. Gazebo
is memory-heavy; if it gets OOM-killed (exit 137), close other apps or run a
single instance at a time (`pkill -f 'gz sim'` to clear leftovers between runs).

That's it. `./run.sh rviz` opens RViz, brings up MoveIt + perception + the TAMP
planner + executor, and ~15 s later auto-sends the goal; you'll see the Panda
animate each planned trajectory (PICK/STACK …), 3 s apart.

Stop any ROS run with **Ctrl-C**.

Tests: `./run.sh` uses the venv; for the unit tests run
`. .venv/bin/activate && python -m pytest tests/unit -q` (the package is an
editable install, so no `PYTHONPATH` is needed).

---

### What each mode does

| Command | ROS? | Window? | Shows |
|---------|------|---------|-------|
| `./run.sh [scene]` | no | no | the symbolic+geometric plan, instantly |
| `./run.sh ros [scene]` | yes | no | the full pipeline; `PLAN OK` per step |
| `./run.sh rviz [scene]` | yes | yes | the arm animating each trajectory in RViz |

`scene` is `stack3` (default) or `replan` (shows closed-loop replanning recovery).

### If RViz shows no robot
Set **Fixed Frame = `panda_link0`** (top-left of RViz), and if there's no arm,
Add → **MotionPlanning**. The trajectories play on `/display_planned_path`.

### Manual / advanced
The longer multi-terminal form and the internals (two-interpreter setup,
Jazzy-specific MoveItPy fixes) are in `docs/ros_integration.md`. The underlying
launch is `pytamp_moveit_bridge tamp_demo.launch.py` (args: `scene`, `rviz`,
`auto_goal`).

### Physics
RViz **animates** the planned (collision-checked, kinematic) trajectories — it
does not simulate gravity/contact. Physics needs a controller manager + Gazebo
(not installed here); the executor already supports `execute:=true` for when they
are present.
