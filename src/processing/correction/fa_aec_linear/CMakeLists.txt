cmake_minimum_required(VERSION 3.8)
project(fa_aec_linear)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(fa_interfaces REQUIRED)

add_executable(fa_aec_linear_node
  src/fa_aec_linear_node.cpp
)

target_include_directories(fa_aec_linear_node PRIVATE
  include
)

ament_target_dependencies(fa_aec_linear_node
  rclcpp
  diagnostic_msgs
  fa_interfaces
)

install(TARGETS fa_aec_linear_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

