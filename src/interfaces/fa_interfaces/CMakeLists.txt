cmake_minimum_required(VERSION 3.8)
project(fa_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

set(message_files
  "msg/AudioFrame.msg"
  "msg/PlaybackDone.msg"
  "msg/VadState.msg"
)

set(service_files
  "srv/ListDevices.srv"
  "srv/SwitchDevice.srv"
  "srv/Record.srv"
  "srv/PlayFile.srv"
  "srv/Speak.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${message_files}
  ${service_files}
  DEPENDENCIES builtin_interfaces std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
