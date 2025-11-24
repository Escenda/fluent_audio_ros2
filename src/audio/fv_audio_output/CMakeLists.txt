cmake_minimum_required(VERSION 3.8)
project(fv_audio_output)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(fv_audio REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ALSA REQUIRED alsa)

include_directories(
  include
  ${ALSA_INCLUDE_DIRS}
)

add_executable(fv_audio_output_node
  src/fv_audio_output_node.cpp
)

ament_target_dependencies(fv_audio_output_node
  rclcpp
  std_msgs
  diagnostic_msgs
  fv_audio
)

 target_include_directories? need etc
