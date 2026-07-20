"""One-command Phase-1 demo.

Starts EVERYTHING: MoveIt + perception + executor + the TAMP planner (run under
the venv interpreter via `prefix`) + optional RViz, and auto-publishes the goal
after a short delay. No second terminal, no manual sourcing of two Pythons.

    ros2 launch pytamp_moveit_bridge tamp_demo.launch.py scene:=stack3 rviz:=true

Args:
  scene:=stack3|replan   which benchmark scene
  rviz:=true|false       open RViz to watch the arm (default false)
  auto_goal:=true|false  auto-send the goal after startup (default true)

The wrapper `./run.sh` sets the environment so you don't have to.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

# Goal per scene (auto-published after startup).
GOALS = {
    "stack3": '{"type":"stack","order":["A","B","C"],"base_surface":"table"}',
    "replan": '{"type":"on_any","object":"block","surfaces":["tray_near","tray_far"]}',
}


def generate_launch_description():
    share = get_package_share_directory("pytamp_moveit_bridge")
    panda_share = get_package_share_directory("panda_moveit_config")

    moveit_config = (
        MoveItConfigsBuilder(robot_name="panda", package_name="panda_moveit_config")
        .robot_description(file_path="config/panda.urdf.xacro")
        .robot_description_semantic(file_path="config/panda.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs())
    md = moveit_config.to_dict()
    md.pop("planning_pipelines", None)
    md["planning_pipelines"] = {"pipeline_names": ["ompl"]}
    md["ompl"] = {**md.get("ompl", {}), "plan_request_params": {
        "planning_pipeline": "ompl", "planner_id": "",
        "planning_time": 5.0, "planning_attempts": 10,
        "max_velocity_scaling_factor": 1.0, "max_acceleration_scaling_factor": 1.0}}

    scene = LaunchConfiguration("scene")
    # venv interpreter for the planner node (it needs pykin/trimesh/numpy<2).
    venv_py = os.environ.get("TAMP_PYTHON", os.path.expanduser("~/Desktop/PDDL/.venv/bin/python3"))
    scene_cfg_dir = os.path.join(share, "config")
    rviz_cfg = os.path.join(panda_share, "config", "motion_planning.rviz")

    # Goal-publishing process per scene (chosen at launch time via OpaqueFunction-free trick:
    # we publish whichever scene's goal the user selected by reading the env the wrapper set).
    scene_name = os.environ.get("TAMP_SCENE", "stack3")
    goal = GOALS.get(scene_name, GOALS["stack3"])

    return LaunchDescription([
        DeclareLaunchArgument("scene", default_value=scene_name),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("auto_goal", default_value="true"),

        Node(package="robot_state_publisher", executable="robot_state_publisher",
             output="screen", parameters=[moveit_config.robot_description]),
        Node(package="joint_state_publisher", executable="joint_state_publisher",
             output="screen", parameters=[moveit_config.robot_description]),

        Node(package="pytamp_moveit_bridge", executable="mock_perception.py",
             name="mock_perception", output="screen", parameters=[{"scene": scene}]),
        Node(package="pytamp_moveit_bridge", executable="action_logger.py",
             name="action_logger", output="screen"),

        Node(package="pytamp_moveit_bridge", executable="moveit_executor.py",
             name="moveit_py", output="screen", parameters=[md]),

        # TAMP planner — run under the venv python via `prefix`.
        Node(package="pytamp_moveit_bridge", executable="tamp_planner_node.py",
             name="tamp_planner", output="screen", prefix=venv_py,
             parameters=[{"scene_config": os.path.join(scene_cfg_dir, f"scene_{scene_name}.yaml")}]),

        Node(package="rviz2", executable="rviz2", name="rviz2", output="screen",
             condition=IfCondition(LaunchConfiguration("rviz")),
             arguments=["-d", rviz_cfg],
             parameters=[moveit_config.robot_description,
                         moveit_config.robot_description_semantic,
                         moveit_config.robot_description_kinematics]),

        # Auto-send the goal once everything is up (~15 s).
        TimerAction(period=15.0, actions=[
            ExecuteProcess(
                condition=IfCondition(LaunchConfiguration("auto_goal")),
                cmd=["ros2", "topic", "pub", "--once", "/planning_goal",
                     "std_msgs/msg/String", f"data: '{goal}'"],
                output="screen")]),
    ])
