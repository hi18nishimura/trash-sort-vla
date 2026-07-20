# ROS 2 Integration (Phase 1, MoveIt 2 execution)

This documents the **live** ROS 2 pipeline that was built and run on **ROS 2 Jazzy**
(Ubuntu 24.04) with **MoveIt 2** already installed system-wide.

```
mock_perception ──/world_state──► tamp_planner ──/task_plan──► action_logger
   (scene fixture)               (PDDL + PyTAMP)            ├─► moveit_executor (MoveItPy / OMPL)
                                  ▲  /planning_goal          └─► (real arm / Gazebo when controllers present)
```

## Packages (`ros2_ws/src/`)

- **pytamp_moveit_bridge** (ament_cmake): the 3 messages (`ObjectState`,
  `WorldState`, `TaskAction`) + 4 Python nodes:
  - `tamp_planner_node.py` — subscribes `/world_state` + `/planning_goal`, runs
    `tamp.tamp_solver.TampSolver` (PDDL+PyTAMP, with geometric replanning),
    publishes refined `TaskAction`s on `/task_plan`.
  - `mock_perception.py` — publishes `/world_state` from a benchmark scene
    (stands in for the reference YOLOv8 `object_state_manager.py`).
  - `moveit_executor.py` — `MoveItPy`; for each `/task_plan` step sets a pose goal
    for `panda_arm` and plans with **OMPL**.
  - `action_logger.py` — logs `/task_plan` (lightweight executor stand-in).
- **panda_moveit_config**, **robot_description** — vendored from the reference,
  build cleanly on Jazzy; provide the Panda URDF/SRDF + OMPL/CHOMP configs.

## Two-interpreter setup (important)

ROS 2 uses **system python3** (`rclpy`). The TAMP geometry deps (`pykin`,
`trimesh`, `numpy<2`) live in `.venv`. So:

- `tamp_planner_node` runs under **`.venv/bin/python3`** with the ROS paths on
  `PYTHONPATH` (it needs both `rclpy` and the geometry deps).
- `moveit_executor` / `mock_perception` / `action_logger` run under **system
  python3** (no geometry deps needed; `MoveItPy` is a system ROS package).

`scripts/run_ros_full.sh` wires this up; `scripts/ros_env.sh` sets the combined env.

## Jazzy-specific fixes discovered while bringing this up

1. **MoveItPy planning pipelines** — `MoveItConfigsBuilder.to_dict()` emits a flat
   `planning_pipelines: [ompl]` (what `move_group` wants), but `MoveItCpp`/MoveItPy
   reads **`planning_pipelines.pipeline_names`** (nested). The launch reshapes it.
2. **Param scoping** — `MoveItPy(node_name="moveit_py")` reads params scoped to the
   node named `moveit_py`, so the launch `Node` must be `name="moveit_py"`; the
   subscriber node uses `use_global_arguments=False` to avoid the `__node` remap.
3. **`plan_request_params`** — without them `plan()` uses an empty pipeline name
   (`No planning pipeline available for name ''`); the executor sets
   `PlanRequestParameters(..., "ompl")` with `planning_pipeline="ompl"`.
4. **Start state** — the zero config self-collides on this Panda URDF
   (`panda_hand-panda_link5`); the executor starts from the SRDF **`ready`** named
   state via `set_start_state(configuration_name="ready")` (manual `RobotState`
   construction segfaulted the MoveItPy bindings).
5. **`mock_perception`** self-locates the repo root so `benchmarks`/`tamp` import
   regardless of launch-time `PYTHONPATH` (or set `TAMP_REPO`).

## Build & run

```bash
cd ros2_ws && source /opt/ros/jazzy/setup.bash
colcon build --packages-select pytamp_moveit_bridge robot_description panda_moveit_config
cd .. && bash scripts/run_ros_full.sh stack3     # or: replan
```

### Verified result (stack3)

```
tamp_planner: TAMP SOLVED in 0 replan(s); publishing 4 step(s) on /task_plan
moveit_executor: [step 1] PICK B  -> (0.500, 0.000, 0.435)   PLAN OK
moveit_executor: [step 2] STACK B -> (0.600, 0.150, 0.511)   PLAN OK
moveit_executor: [step 3] PICK A  -> (0.400,-0.150, 0.435)   PLAN OK
moveit_executor: [step 4] STACK A -> (0.600, 0.150, 0.562)   PLAN OK
```

### Verified result (replan) — closed-loop recovery over ROS

```
tamp_planner: TAMP SOLVED in 1 replan(s); falsified=[('placeable','block','tray_far')]
  PICK block -> ... ; PLACE block on tray_near (reachable) ; OMPL PLAN OK
```

Note: that transcript came from a run where the planner explored the far tray
first. Which disjunct the satisficing planner tries first is an internal
implementation detail and is deliberately not relied upon. The invariant the
benchmark (and its tests) assert is the landing tray, not the replan count —
`replans=0` runs are equally correct.

## Not run here (environment limits)

- **Trajectory execution / Gazebo physics**: `moveit_fake_controller_manager` is
  not installed and the box is headless, so we validate at the **motion-planning**
  level (OMPL trajectories). With a controller manager + `ros_gz` spawn, the same
  `moveit_executor` executes the trajectories (`execute:=true`).
