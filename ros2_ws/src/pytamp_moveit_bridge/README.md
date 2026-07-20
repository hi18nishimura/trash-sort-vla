# pytamp_moveit_bridge (TAMP wiring)

Wires the tested, ROS-free `tamp/` core into the reference repo's ROS 2 node
structure. This package **replaces the two stubs** in
[`vishal-finch/TAMP`](https://github.com/vishal-finch/TAMP):

| Reference stub | Replaced by |
|----------------|-------------|
| `src/.../planner_node.py` (MCTS placeholder, mocks a plan) | `scripts/tamp_planner_node.py` (real PDDL+PyTAMP loop) |
| `src/.../scene_interface.py` (PyTAMP calls commented out) | `tamp/geometry/scene_interface.py` + `tamp/tamp_solver.py` |

The reference's working nodes are reused unchanged:
- `object_state_manager.py` (YOLOv8-OBB + depth → `/world_state`)
- `action_executor.py` (MoveItPy pick/place, consumes `/task_plan`)
- `WorldState` / `ObjectState` / `TaskAction` messages

## Topology

```
/world_state ──► tamp_planner_node ──► /task_plan ──► action_executor (MoveItPy)
/planning_goal ─┘   (TampSolver)
```

## Build & run (ROS 2 Humble + MoveIt 2)

```bash
# make the tamp core importable to the node (one option)
export PYTHONPATH=$PYTHONPATH:/home/manas-reddy/Desktop/PDDL

cd ros2_ws
colcon build --packages-select pytamp_moveit_bridge yolov8_obb_msgs
source install/setup.bash

# bring up MoveIt + perception + the TAMP planner (reuses reference launch)
ros2 launch pytamp_moveit_bridge integration.launch.py use_mock_camera:=true

# in another shell, give the TAMP planner a goal
ros2 run pytamp_moveit_bridge tamp_planner_node.py \
    --ros-args -p scene_config:=$(pwd)/src/pytamp_moveit_bridge/config/scene_tabletop.yaml
ros2 topic pub --once /planning_goal std_msgs/String \
    "data: '{\"type\":\"stack\",\"order\":[\"A\",\"B\",\"C\"],\"base_surface\":\"table\"}'"

ros2 topic echo /task_plan      # observe the refined plan steps
```

`scene_config` provides the support surfaces (tables/trays/bins) that are not
present in the perception `/world_state` message. See `config/scene_tabletop.yaml`.

## Note on dependencies

`tamp/geometry` needs `pykin` + `trimesh` + `numpy<2`. On a ROS 2 machine,
install these into the same Python environment the node runs under. The robot
URDF used for IK is pykin's bundled Panda by default; to use the workspace's
`robot_description/urdf/panda.urdf`, pass it to `RobotModel(urdf=...)`.
