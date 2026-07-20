"""Capstone: YOLO-perceived pick-and-place in Gazebo physics.

  gz (camera + objects) -> YOLOv8-COCO -> /world_state -> TAMP (PDDL+PyTAMP)
       -> /task_plan -> MoveIt pick/place -> Gazebo arm -> grasp_manager attach

  ros2 launch pytamp_moveit_bridge tamp_yolo_pnp.launch.py gui:=true

Runs YOLO + TAMP planner under the venv interpreter (ultralytics/pykin); MoveIt,
grasp_manager, bridges under system python. Driven by ./run.sh yolo.
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

GOAL = '{"type":"gather","surface":"tray"}'


def generate_launch_description():
    bridge = get_package_share_directory("pytamp_moveit_bridge")
    rd = get_package_share_directory("robot_description")
    res_path = os.path.dirname(rd)
    venv_py = os.environ.get("TAMP_PYTHON", os.path.expanduser("~/Desktop/PDDL/.venv/bin/python3"))
    world = os.path.join(bridge, "worlds", "tabletop_pnp.sdf")
    gz_urdf = xacro.process_file(os.path.join(bridge, "description", "panda_gazebo.urdf.xacro")).toxml()

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
        "planning_attempts": 10, "max_velocity_scaling_factor": 1.0,
        "max_acceleration_scaling_factor": 1.0}}
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
        DeclareLaunchArgument("gui", default_value="true"),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", res_path),

        IncludeLaunchDescription(PythonLaunchDescriptionSource(gz_launch),
            condition=IfCondition(gui),
            launch_arguments={"gz_args": f"-r -v1 {world}"}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(gz_launch),
            condition=UnlessCondition(gui),
            launch_arguments={"gz_args": f"-r -s -v1 {world}"}.items()),

        # bridges: clock, camera (rgb+depth+info), and model poses for grasp_manager
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"]),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             arguments=["/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
                        "/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
                        "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"]),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             arguments=["/world/tabletop/pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"],
             remappings=[("/world/tabletop/pose/info", "/model_poses")]),

        Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
             parameters=[{"robot_description": gz_urdf, "use_sim_time": True}]),
        spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[arm])),
        RegisterEventHandler(OnProcessExit(target_action=arm, on_exit=[hand])),

        # perception (venv): YOLOv8-COCO -> /world_state (camera fixed in world)
        Node(package="pytamp_moveit_bridge", executable="yolo_perception.py",
             name="yolo_perception", output="screen", prefix=venv_py,
             parameters=[{"base_frame": "world", "cam_xyz": [0.0, 0.0, 1.0],
                          "cam_rpy": [0.0, 0.82, 0.0], "use_sim_time": True}]),

        # kinematic grasp-attach (system python: rclpy+tf2)
        Node(package="pytamp_moveit_bridge", executable="grasp_manager.py",
             name="grasp_manager", output="screen",
             parameters=[{"world": "tabletop", "gripper_frame": "panda_hand",
                          "base_frame": "world", "use_sim_time": True}]),

        # MoveIt pick/place executor (env enables execute + pick-place sequence)
        Node(package="pytamp_moveit_bridge", executable="moveit_executor.py",
             name="moveit_py", output="screen", parameters=[md],
             additional_env={"TAMP_EXECUTE": "1", "TAMP_START_CURRENT": "1",
                             "TAMP_PICKPLACE": "1", "TAMP_DWELL": "0.0",
                             "TAMP_SCENE_CONFIG": os.path.join(bridge, "config", "scene_pnp.yaml")}),

        # TAMP planner (venv): consumes /world_state, default surface = workspace
        Node(package="pytamp_moveit_bridge", executable="tamp_planner_node.py",
             name="tamp_planner", output="screen", prefix=venv_py,
             parameters=[{"scene_config": os.path.join(bridge, "config", "scene_pnp.yaml"),
                          "default_surface": "workspace", "use_sim_time": True}]),

        # auto-send the goal once perception + planning + controllers are up
        TimerAction(period=30.0, actions=[
            ExecuteProcess(cmd=["ros2", "topic", "pub", "--once", "/planning_goal",
                                "std_msgs/msg/String", f"data: '{GOAL}'"], output="screen")]),
    ])
