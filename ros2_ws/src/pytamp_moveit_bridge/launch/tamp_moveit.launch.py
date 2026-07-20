"""Full Phase-1 stack with MoveIt 2 execution:

  mock_perception -> /world_state -> tamp_planner (PDDL+PyTAMP) -> /task_plan
                                                                      |
                                              moveit_executor (MoveItPy / OMPL)

  ros2 launch pytamp_moveit_bridge tamp_moveit.launch.py scene:=stack3

NOTE: tamp_planner_node must run under the venv python (TAMP geometry deps).
This launch starts MoveIt + perception + executor; start the planner separately
with the venv interpreter (see docs/ros_integration.md / scripts/run_ros_full.sh).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder(robot_name="panda", package_name="panda_moveit_config")
        .robot_description(file_path="config/panda.urdf.xacro")
        .robot_description_semantic(file_path="config/panda.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    md = moveit_config.to_dict()
    # MoveItCpp (MoveItPy) reads pipeline names from nested
    # planning_pipelines.pipeline_names; to_dict() emits the flat list that
    # move_group wants, so reshape it here (Jazzy requirement).
    md.pop("planning_pipelines", None)
    md["planning_pipelines"] = {"pipeline_names": ["ompl"]}
    # Default plan-request params for MoveItPy (read under the "ompl" namespace).
    md["ompl"] = {**md.get("ompl", {}), "plan_request_params": {
        "planning_pipeline": "ompl", "planner_id": "",
        "planning_time": 5.0, "planning_attempts": 10,
        "max_velocity_scaling_factor": 1.0, "max_acceleration_scaling_factor": 1.0}}
    scene = LaunchConfiguration("scene")

    rviz_cfg = os.path.join(
        get_package_share_directory("panda_moveit_config"), "config", "motion_planning.rviz")

    return LaunchDescription([
        DeclareLaunchArgument("scene", default_value="stack3"),
        DeclareLaunchArgument("rviz", default_value="false",
                              description="open RViz to watch the arm plan/move"),

        # RViz (MotionPlanning display animates /display_planned_path)
        Node(package="rviz2", executable="rviz2", name="rviz2", output="screen",
             condition=IfCondition(LaunchConfiguration("rviz")),
             arguments=["-d", rviz_cfg],
             parameters=[moveit_config.robot_description,
                         moveit_config.robot_description_semantic,
                         moveit_config.robot_description_kinematics]),

        # robot state publisher (TF from the URDF)
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             output="screen", parameters=[moveit_config.robot_description]),

        # joint_state_publisher: provides /joint_states so MoveItPy has a valid
        # *current* robot state to plan FROM (no real controllers in this demo).
        Node(package="joint_state_publisher", executable="joint_state_publisher",
             output="screen", parameters=[moveit_config.robot_description]),

        # perception (mock) + plan logger
        Node(package="pytamp_moveit_bridge", executable="mock_perception.py",
             name="mock_perception", output="screen", parameters=[{"scene": scene}]),
        Node(package="pytamp_moveit_bridge", executable="action_logger.py",
             name="action_logger", output="screen"),

        # MoveIt executor (plans OMPL trajectories to /task_plan poses). Named
        # "moveit_py" so the config params scope to MoveItPy's internal node.
        Node(package="pytamp_moveit_bridge", executable="moveit_executor.py",
             name="moveit_py", output="screen", parameters=[md]),
    ])
