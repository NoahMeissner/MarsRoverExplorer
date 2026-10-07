cmake_minimum_required(VERSION 3.8)
project(cave_explorer_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Artefact.msg"
  "msg/ArtefactArray.msg"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

ament_package()
