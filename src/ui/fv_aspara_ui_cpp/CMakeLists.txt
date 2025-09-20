cmake_minimum_required(VERSION 3.8)
project(fv_aspara_ui_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(fv_msgs REQUIRED)
find_package(fluent_lib REQUIRED)

set(FV_OPENCV_LIBS)
if(TARGET opencv_core)
  list(APPEND FV_OPENCV_LIBS opencv_core opencv_imgproc)
else()
  list(APPEND FV_OPENCV_LIBS ${OpenCV_LIBS})
endif()

add_executable(fv_aspara_ui_cpp_node
  src/fv_aspara_ui_cpp_node.cpp
)

ament_target_dependencies(fv_aspara_ui_cpp_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  geometry_msgs
  fv_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

target_include_directories(fv_aspara_ui_cpp_node
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(fv_aspara_ui_cpp_node
  ${FV_OPENCV_LIBS}
  tiff
  curl
  fluent_lib::fluent_lib
)

target_compile_features(fv_aspara_ui_cpp_node PUBLIC cxx_std_17)

install(TARGETS fv_aspara_ui_cpp_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
