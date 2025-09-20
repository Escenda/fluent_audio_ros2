cmake_minimum_required(VERSION 3.8)
project(fv_aspara_points)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(fv_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)
find_package(pcl_conversions REQUIRED)

add_executable(fv_aspara_points_node
  src/fv_aspara_points_node.cpp
)

ament_target_dependencies(fv_aspara_points_node
  rclcpp sensor_msgs geometry_msgs visualization_msgs builtin_interfaces
  tf2 tf2_ros tf2_geometry_msgs fv_msgs PCL pcl_conversions
)

target_include_directories(fv_aspara_points_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(TARGETS fv_aspara_points_node
  DESTINATION lib/${PROJECT_NAME}
)

## No package-local config directory for now

ament_package()
