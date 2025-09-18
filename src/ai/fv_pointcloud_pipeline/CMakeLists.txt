cmake_minimum_required(VERSION 3.8)
project(fv_pointcloud_pipeline)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(fluent_lib REQUIRED)
find_package(fv_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(yaml-cpp REQUIRED)

set(FV_OPENCV_LIBS)
if(TARGET opencv_core)
  list(APPEND FV_OPENCV_LIBS opencv_core opencv_imgproc)
else()
  list(APPEND FV_OPENCV_LIBS ${OpenCV_LIBS})
endif()

add_executable(fv_pointcloud_pipeline_node
  src/fv_pointcloud_pipeline_node.cpp
)

ament_target_dependencies(fv_pointcloud_pipeline_node
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  pcl_conversions
  fluent_lib
  fv_msgs
)

target_include_directories(fv_pointcloud_pipeline_node
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(fv_pointcloud_pipeline_node
  ${FV_OPENCV_LIBS}
  ${PCL_LIBRARIES}
  yaml-cpp
  fluent_lib::fluent_lib
  tiff
  curl
)

target_compile_features(fv_pointcloud_pipeline_node PUBLIC cxx_std_17)

install(TARGETS fv_pointcloud_pipeline_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
