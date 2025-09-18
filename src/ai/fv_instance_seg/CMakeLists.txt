cmake_minimum_required(VERSION 3.8)
project(fv_instance_seg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(OpenVINO REQUIRED)
find_package(fluent_lib REQUIRED)

include_directories(include)
include_directories(${fluent_lib_INCLUDE_DIRS})

# Prefer linking only the required OpenCV modules to avoid pulling
# in optional dependencies (e.g., GDAL, libtiff, libcurl) via a
# broad OpenCV_LIBS list present in some environments.
set(FV_OPENCV_LIBS)
if(TARGET opencv_core)
  list(APPEND FV_OPENCV_LIBS opencv_core opencv_imgproc)
else()
  # Fallback for environments without imported OpenCV targets
  list(APPEND FV_OPENCV_LIBS ${OpenCV_LIBS})
endif()

add_library(fv_instance_seg_backends
  src/backends/noop_inferencer.cpp
  src/backends/ov_yolo_seg.cpp
)
target_link_libraries(fv_instance_seg_backends
  ${FV_OPENCV_LIBS}
  openvino::runtime
)
ament_target_dependencies(fv_instance_seg_backends
  rclcpp sensor_msgs vision_msgs cv_bridge
)

add_executable(fv_instance_seg_node
  src/fv_instance_seg_node.cpp
)
target_link_libraries(fv_instance_seg_node
  fv_instance_seg_backends
  ${FV_OPENCV_LIBS}
  tiff
  curl
  fluent_lib::fluent_lib
)
ament_target_dependencies(fv_instance_seg_node
  rclcpp sensor_msgs vision_msgs cv_bridge fluent_lib
)

install(TARGETS fv_instance_seg_backends fv_instance_seg_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
