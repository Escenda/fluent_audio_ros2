cmake_minimum_required(VERSION 3.8)
project(fa_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

set(message_files
  "msg/AsrResult.msg"
  "msg/AudioFrame.msg"
  "msg/CqtFrame.msg"
  "msg/EncodedAudioChunk.msg"
  "msg/LoudnessFrame.msg"
  "msg/LogMelFrame.msg"
  "msg/MfccFrame.msg"
  "msg/OnsetFrame.msg"
  "msg/PlaybackDone.msg"
  "msg/PitchFrame.msg"
  "msg/StftFrame.msg"
  "msg/TempoFrame.msg"
  "msg/TurnContext.msg"
  "msg/TurnEnd.msg"
  "msg/VadState.msg"
  "msg/WakeWordResult.msg"
)

set(service_files
  "srv/ListDevices.srv"
  "srv/SwitchDevice.srv"
  "srv/Record.srv"
  "srv/Speak.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${message_files}
  ${service_files}
  DEPENDENCIES builtin_interfaces std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)

  # Keep package tests isolated from ambient pytest plugins in ROS images.
  set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "1")
  ament_add_pytest_test(${PROJECT_NAME}_pytest test
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30
  )
endif()

ament_package()
