cmake_minimum_required(VERSION 3.10)
project(fv_detection_fusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(fv_msgs REQUIRED)

add_executable(fv_detection_fusion_node
  src/fv_detection_fusion_node.cpp
)
ament_target_dependencies(fv_detection_fusion_node
  rclcpp
  sensor_msgs
  std_msgs
  vision_msgs
  message_filters
  fv_msgs
)

target_compile_features(fv_detection_fusion_node PUBLIC cxx_std_17)

install(TARGETS fv_detection_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
