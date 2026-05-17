cmake_minimum_required(VERSION 3.8)
project(fa_out)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(fa_interfaces REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ALSA REQUIRED alsa)

add_executable(fa_out_node
  src/fa_out_node.cpp
)

ament_target_dependencies(fa_out_node
  rclcpp
  std_msgs
  diagnostic_msgs
  fa_interfaces
)

target_include_directories(fa_out_node PRIVATE
  include
  ${ALSA_INCLUDE_DIRS}
)
target_link_libraries(fa_out_node ${ALSA_LIBRARIES})

install(TARGETS fa_out_node
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
