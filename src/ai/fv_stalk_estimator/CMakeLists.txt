cmake_minimum_required(VERSION 3.8)
project(fv_stalk_estimator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(fluent_lib REQUIRED)
find_package(fv_msgs REQUIRED)

add_executable(fv_stalk_estimator_node
  src/fv_stalk_estimator_node.cpp
)

ament_target_dependencies(fv_stalk_estimator_node
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  visualization_msgs
  pcl_conversions
  fluent_lib
  fv_msgs
)

target_include_directories(fv_stalk_estimator_node
  PRIVATE
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(fv_stalk_estimator_node
  ${PCL_LIBRARIES}
  fluent_lib::fluent_lib
)

target_compile_features(fv_stalk_estimator_node PUBLIC cxx_std_17)

install(TARGETS fv_stalk_estimator_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
