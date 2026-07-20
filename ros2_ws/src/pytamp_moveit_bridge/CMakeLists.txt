cmake_minimum_required(VERSION 3.8)
project(pytamp_moveit_bridge)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectState.msg"
  "msg/WorldState.msg"
  "msg/TaskAction.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# Python nodes (the TAMP brain + a mock perception source + a plan logger).
install(PROGRAMS
  scripts/tamp_planner_node.py
  scripts/mock_perception.py
  scripts/action_logger.py
  scripts/moveit_executor.py
  scripts/yolo_perception.py
  scripts/grasp_manager.py
  scripts/nl_goal_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config description worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
