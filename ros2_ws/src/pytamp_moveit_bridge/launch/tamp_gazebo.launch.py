"""One-command Gazebo physics demo.

Spawns the Panda in Gazebo (gz-sim8 / gz_ros2_control), brings up the arm
controller, and runs the full TAMP pipeline so the arm **physically executes**
the planned trajectories under gravity:

    mock_perception -> /world_state -> tamp_planner (PDDL+PyTAMP) -> /task_plan
                                                                       |
                              moveit_executor (MoveItPy/OMPL, execute:=true)
                                                                       |
                          panda_arm_controller (joint_trajectory_controller) -> Gazebo

    ros2 launch pytamp_moveit_bridge tamp_gazebo.launch.py scene:=stack3 gui:=true

Use the ./run.sh gazebo wrapper (sets env + venv interpreter for the planner).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, RegisterEventHandler,
                            SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
import xacro

GOALS = {
    "stack3": '{"type":"stack","order":["A","B","C"],"base_surface":"table"}',
    "replan": '{"type":"on_any","object":"block","surfaces":["tray_near","tray_far"]}',
}


def generate_launch_description():
    bridge = get_package_share_directory("pytamp_moveit_bridge")
    panda = get_package_share_directory("panda_moveit_config")
    rd = get_package_share_directory("robot_description")
    res_path = os.path.dirname(rd)  # so package://robot_description/... resolves in Gazebo

    scene_name = os.environ.get("TAMP_SCENE", "stack3")
    goal = GOALS.get(scene_name, GOALS["stack3"])
    venv_py = os.environ.get("TAMP_PYTHON", os.path.expanduser("~/Desktop/PDDL/.venv/bin/python3"))

    gz_urdf = xacro.process_file(
        os.path.join(bridge, "description", "panda_gazebo.urdf.xacro")).toxml()

    # MoveIt config incl. trajectory execution (moveit_simple_controller_manager).
    moveit_config = (
        MoveItConfigsBuilder(robot_name="panda", package_name="panda_moveit_config")
        .robot_description(file_path="config/panda.urdf.xacro")
        .robot_description_semantic(file_path="config/panda.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs())
    md = moveit_config.to_dict()
    md.pop("planning_pipelines", None)
    md["planning_pipelines"] = {"pipeline_names": ["ompl"]}
    md["ompl"] = {**md.get("ompl", {}), "plan_request_params": {
        "planning_pipeline": "ompl", "planner_id": "",
        "planning_time": 5.0, "planning_attempts": 10,
        "max_velocity_scaling_factor": 1.0, "max_acceleration_scaling_factor": 1.0}}
    md["use_sim_time"] = True

    gz_launch = os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
    gui = LaunchConfiguration("gui")

    jsb = Node(package="controller_manager", executable="spawner", output="screen",
               arguments=["joint_state_broadcaster"])
    arm = Node(package="controller_manager", executable="spawner", output="screen",
               arguments=["panda_arm_controller"])
    hand = Node(package="controller_manager", executable="spawner", output="screen",
               arguments=["panda_hand_controller"])

    spawn = Node(package="ros_gz_sim", executable="create", output="screen",
                 arguments=["-topic", "/robot_description", "-name", "panda", "-z", "0.0"])

    return LaunchDescription([
        DeclareLaunchArgument("scene", default_value=scene_name),
        DeclareLaunchArgument("gui", default_value="true",
                              description="Gazebo GUI (false = headless server)"),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", res_path),

        # Gazebo: GUI by default, headless server (-s) when gui:=false
        IncludeLaunchDescription(PythonLaunchDescriptionSource(gz_launch),
            condition=IfCondition(gui),
            launch_arguments={"gz_args": "-r -v1 empty.sdf"}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(gz_launch),
            condition=UnlessCondition(gui),
            launch_arguments={"gz_args": "-r -s -v1 empty.sdf"}.items()),

        # /clock bridge so all ROS nodes share sim time
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"]),

        Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
             parameters=[{"robot_description": gz_urdf, "use_sim_time": True}]),
        spawn,

        # Spawn controllers once the robot entity exists.
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[arm])),
        RegisterEventHandler(OnProcessExit(target_action=arm, on_exit=[hand])),

        # Perception + plan logger
        Node(package="pytamp_moveit_bridge", executable="mock_perception.py",
             name="mock_perception", output="screen",
             parameters=[{"scene": LaunchConfiguration("scene"), "use_sim_time": True}]),
        Node(package="pytamp_moveit_bridge", executable="action_logger.py",
             name="action_logger", output="screen", parameters=[{"use_sim_time": True}]),

        # MoveIt executor — EXECUTES trajectories on the Gazebo controller.
        # execute/start_from_current come via env (the subscriber node ignores
        # node-scoped launch params; see moveit_executor.py).
        Node(package="pytamp_moveit_bridge", executable="moveit_executor.py",
             name="moveit_py", output="screen", parameters=[md],
             additional_env={"TAMP_EXECUTE": "1", "TAMP_START_CURRENT": "1", "TAMP_DWELL": "0.0"}),

        # TAMP planner (venv interpreter).
        Node(package="pytamp_moveit_bridge", executable="tamp_planner_node.py",
             name="tamp_planner", output="screen", prefix=venv_py,
             parameters=[{"scene_config": os.path.join(bridge, "config", f"scene_{scene_name}.yaml"),
                          "use_sim_time": True}]),

        # Auto-send the goal once everything is up (Gazebo + controllers take longer).
        TimerAction(period=22.0, actions=[
            ExecuteProcess(
                cmd=["ros2", "topic", "pub", "--once", "/planning_goal",
                     "std_msgs/msg/String", f"data: '{goal}'"],
                output="screen")]),
    ])
