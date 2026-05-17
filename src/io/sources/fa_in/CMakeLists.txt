cmake_minimum_required(VERSION 3.8)
project(fa_in)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_srvs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(fa_interfaces REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ALSA REQUIRED alsa)

add_library(fa_in_backends
  src/backends/alsa_capture_backend.cpp
)

target_include_directories(fa_in_backends
  PUBLIC
  include
  PRIVATE
  ${ALSA_INCLUDE_DIRS}
)
target_link_libraries(fa_in_backends PUBLIC ${ALSA_LIBRARIES})

add_executable(fa_in_node
  src/fa_in_node.cpp
)

ament_target_dependencies(fa_in_node
  rclcpp
  std_srvs
  diagnostic_msgs
  fa_interfaces
)

target_include_directories(fa_in_node PRIVATE
  include
)
target_link_libraries(fa_in_node fa_in_backends)

install(TARGETS fa_in_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS fa_in_backends
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
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
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  # Keep package tests isolated from ambient pytest plugins in ROS images.
  set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "1")
  ament_add_pytest_test(${PROJECT_NAME}_pytest test
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30
  )

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
