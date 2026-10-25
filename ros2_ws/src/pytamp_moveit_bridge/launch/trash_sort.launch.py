"""Camera-free verification of the pick-and-place manipulation+grasp chain:
mock_perception (scene_pnp) -> TAMP -> MoveIt pick/place -> Gazebo + grasp_manager.
Same as tamp_yolo_pnp but with mock perception instead of the camera (so it runs
headless without GPU rendering). Used to verify grasp-attach + placement.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, RegisterEventHandler,
                            SetEnvironmentVariable, TimerAction,
                            OpaqueFunction)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
import xacro

GOAL = '{"type":"gather","surface":"tray"}'


import json


def _send_goal(context):
    """Send one sorting goal: put <item> into <bin>."""
    item = LaunchConfiguration("item").perform(context)
    bin_id = LaunchConfiguration("bin").perform(context)
    if item == "none":
        return []
    goal = json.dumps({"type": "place_class", "class": item, "surface": bin_id},
                      separators=(",", ":"))
    return [ExecuteProcess(cmd=["ros2", "topic", "pub", "--once", "/planning_goal",
                                "std_msgs/msg/String", f"data: '{goal}'"], output="screen")]


def generate_launch_description():
    bridge = get_package_share_directory("pytamp_moveit_bridge")
    rd = get_package_share_directory("robot_description")
    venv_py = os.environ.get("TAMP_PYTHON", os.path.expanduser("~/Desktop/PDDL/.venv/bin/python3"))
    world = os.path.join(bridge, "worlds", "trash_sort.sdf")
    gz_urdf = xacro.process_file(os.path.join(bridge, "description", "panda_gazebo_trash.urdf.xacro")).toxml()

    mc = (MoveItConfigsBuilder(robot_name="panda", package_name="panda_moveit_config")
          .robot_description(file_path="config/panda.urdf.xacro")
          .robot_description_semantic(file_path="config/panda.srdf")
          .robot_description_kinematics(file_path="config/kinematics.yaml")
          .trajectory_execution(file_path="config/moveit_controllers.yaml")
          .planning_pipelines(pipelines=["ompl"]).to_moveit_configs())
    md = mc.to_dict()
    md.pop("planning_pipelines", None)
    md["planning_pipelines"] = {"pipeline_names": ["ompl"]}
    md["ompl"] = {**md.get("ompl", {}), "plan_request_params": {
        "planning_pipeline": "ompl", "planner_id": "", "planning_time": 5.0,
        "planning_attempts": 10, "max_velocity_scaling_factor": 0.3,
        "max_acceleration_scaling_factor": 0.3}}
    md["use_sim_time"] = True

    gz_launch = os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
    gui = LaunchConfiguration("gui")
    spawn = Node(package="ros_gz_sim", executable="create", output="screen",
                 arguments=["-topic", "/robot_description", "-name", "panda", "-z", "0.0"])
    jsb = Node(package="controller_manager", executable="spawner", output="screen",
               arguments=["joint_state_broadcaster"])
    arm = Node(package="controller_manager", executable="spawner", output="screen",
               arguments=["panda_arm_controller"])
    hand = Node(package="controller_manager", executable="spawner", output="screen",
               arguments=["panda_hand_controller"])

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("item", default_value="can"),
        DeclareLaunchArgument("bin", default_value="bin_blue"),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.path.dirname(rd)),

        # Physics engine: bullet-featherstone, NOT gz's default DART. DART silently
        # drops mimic constraints AND locks the finger joint at its 0.04 limit after
        # the first grasp (every command afterward is ignored while the controller
        # reports success — the 2nd cup could never be picked). bullet-featherstone
        # handles the articulated finger joints without that lock.
        IncludeLaunchDescription(PythonLaunchDescriptionSource(gz_launch),
            condition=IfCondition(gui),
            launch_arguments={"gz_args":
                f"-r -v1 --physics-engine gz-physics-bullet-featherstone-plugin {world}"}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(gz_launch),
            condition=UnlessCondition(gui),
            launch_arguments={"gz_args":
                f"-r -s -v1 --physics-engine gz-physics-bullet-featherstone-plugin {world}"}.items()),

        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"]),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             arguments=["/overhead_camera/image@sensor_msgs/msg/Image[gz.msgs.Image"]),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             arguments=["/world/tabletop/pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"],
             remappings=[("/world/tabletop/pose/info", "/model_poses")]),

        Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
             parameters=[{"robot_description": gz_urdf, "use_sim_time": True}]),
        spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[arm])),
        RegisterEventHandler(OnProcessExit(target_action=arm, on_exit=[hand])),

        # mock perception (scene_pnp) instead of the camera/YOLO
        # live_poses: follow the cups' real Gazebo poses (via the /model_poses bridge)
        # rather than republishing their start positions. Without this a placed cup stays
        # a PHANTOM at its old spot in the planning scene while the real one on the tray
        # is invisible to OMPL — the arm sweeps through and knocks it off the tray.
        Node(package="pytamp_moveit_bridge", executable="mock_perception.py",
             name="mock_perception", output="screen", prefix=venv_py,
             # live_poses MUST stay True: with static poses, a placed cup_1 is a
             # phantom at its old table spot while the REAL cup on the tray is
             # invisible — and the placement sampler tries the tray CENTRE first, so
             # cup_2's descent is then validated as free at the exact spot cup_1
             # occupies and the arm drives the welded cup into it (multi-metre fling).
             parameters=[{"scene": "trash", "use_sim_time": True, "live_poses": True,
                          "model_map": ["can_1:can", "bottle_1:bottle", "paper_1:paper"]}]),

        Node(package="pytamp_moveit_bridge", executable="grasp_manager.py",
             name="grasp_manager", output="screen",
             parameters=[{"world": "tabletop", "gripper_frame": "panda_hand",
                          "base_frame": "world", "use_sim_time": True,
                          "graspable_models": ["can", "bottle", "paper"]}]),
        Node(package="pytamp_moveit_bridge", executable="moveit_executor.py",
             name="moveit_py", output="screen", parameters=[md],
             additional_env={"TAMP_EXECUTE": "1", "TAMP_START_CURRENT": "1",
                             "TAMP_PICKPLACE": "1", "TAMP_DWELL": "0.0", "TAMP_VEL_SCALE": "0.3",
                             "TAMP_SCENE_CONFIG": os.path.join(bridge, "config", "scene_trash.yaml")}),
        Node(package="pytamp_moveit_bridge", executable="tamp_planner_node.py",
             name="tamp_planner", output="screen", prefix=venv_py,
             parameters=[{"scene_config": os.path.join(bridge, "config", "scene_trash.yaml"),
                          "default_surface": "workspace", "use_sim_time": True}]),

        TimerAction(period=26.0, actions=[OpaqueFunction(function=_send_goal)]),
    ])
