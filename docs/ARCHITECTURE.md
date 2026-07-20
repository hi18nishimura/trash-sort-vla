# Architecture

## Three-layer pipeline (matches the reference images)

```
┌─ PERCEPTION ────────────────────────────────────────────────────────────┐
│ Camera(RGB+D) → YOLOv8 (COCO) → depth back-projection (2D→3D, TF) → /world_state │
└──────────────────────────────────────────────────────────────────────────┘
                                   │  WorldState (ObjectState[])
┌─ PLANNING ────────────────────────────────────────────────────────────────┐
│ /world_state → SceneInterface (PyTAMP world model)                          │
│                  │                                                          │
│        object_reasoner → init facts (on/clear/stacked_on)                   │
│                  │                                                          │
│        problem_generator → unified_planning.Problem (bridge preds optimistic)│
│                  │                                                          │
│   ┌────────► symbolic_planner (Fast Downward) ── symbolic plan ──┐          │
│   │                                                              ▼          │
│   │   GeometricRefiner: grasp sampling + DLS IK + collision (per action)    │
│   │                                                              │          │
│   └──── falsify failed bridge predicate ◄── geometric failure ───┘          │
│                  │  geometrically-feasible refined plan                     │
└──────────────────┼──────────────────────────────────────────────────────────┘
                   │  TaskAction[] (action_type, object_id, target_pose, pre/eff)
┌─ EXECUTION ──────┴────────────────────────────────────────────────────────┐
│ /task_plan → ActionExecutor → MoveIt 2 (OMPL plan + CHOMP smooth) → robot  │
└──────────────────────────────────────────────────────────────────────────┘
```

## The TAMP loop (`tamp/tamp_solver.py`)

The crux of the system and the part the reference repo never built.

1. **Generate** a PDDL problem from the world state. Symbolic facts `on`,
   `clear`, `stacked_on` are derived from geometry by `object_reasoner`. The
   geometric **bridge predicates** `reachable / graspable / stackable /
   placeable` are seeded **optimistically True**.
2. **Plan** symbolically (Fast Downward) → ordered `pick/place/stack/unstack`.
3. **Refine** each action geometrically against a *shadow scene* (which mutates
   object poses as the simulated plan proceeds, so a later `stack` sees the
   bottom object where the prior `place` left it):
   - `pick/unstack`: sample top/side grasps from the object OBB; require IK for
     both grasp and pre-grasp poses, and a collision-free gripper approach.
   - `place/stack`: compute a top-down EEF pose over the destination; require IK
     and a collision-free placement footprint.
4. On the first infeasible action, **falsify** the responsible bridge predicate
   (e.g. `(not (placeable block tray_far))`) and **go to step 1**. If the same
   predicate fails twice, or the problem becomes unsolvable, report failure.

This realizes "constraint-aware planning": symbolic search is steered by
geometric feasibility without hard-coding geometry into the PDDL model.

## Why the geometry layer is hand-rolled on pykin

`pytamp` provides scene/grasp/motion primitives but pulls in `pygraphviz`
(system `graphviz-dev`, used only for MCTS tree viz). We instead use its core
deps directly: `pykin` for FK/Jacobian and `trimesh` for collision. `pykin`'s
own IK is broken on Python 3.12 / NumPy ≥2 (NaN quaternions, removed
`collections.Iterable`), so:

- `tamp/geometry/_compat.py` restores the `collections` ABC aliases pre-import.
- IK is reimplemented as damped least squares on pykin's working FK + Jacobian,
  with joint-limit clamping and random restarts
  (`tamp/geometry/robot_model.py`).
- NumPy is pinned `<2`.

The robot is pykin's bundled **Franka Panda** (`urdf/panda/panda.urdf`, base
`panda_link_0`, EEF `right_gripper`), matching the reference repo's arm.

## Keeping the two domain definitions in sync

The domain exists as a canonical `pddl/domains/tabletop.pddl` (portable,
inspectable) and a runtime `unified_planning` model in
`tamp/pddl/domain_model.py`. Any predicate/action change must touch both, plus
the seeding logic in `problem_generator.py`. `GEOMETRIC_FLUENTS` in
`domain_model.py` is the canonical list of bridge predicates.

## ROS 2 contract (Phase 1 execution)

The pure-Python `tamp/types.py` dataclasses mirror the reference messages so the
bridge converts 1:1:

| ROS message | dataclass |
|-------------|-----------|
| `pytamp_moveit_bridge/WorldState` (`ObjectState[]`) | `WorldState` |
| `pytamp_moveit_bridge/ObjectState` | `ObjectState` |
| `pytamp_moveit_bridge/TaskAction` | `TaskAction` |

Quaternions use `(w, x, y, z)` ordering to match the reference's
`scene_interface.py`.
