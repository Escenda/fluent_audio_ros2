cmake_minimum_required(VERSION 3.8)
project(fa_vad)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(fa_interfaces REQUIRED)

ament_python_install_package(fa_vad_py PACKAGE_DIR fa_vad_py)

install(PROGRAMS
  scripts/fa_vad_node
  scripts/silero_vad_worker
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(
  rclpy
  std_msgs
  fa_interfaces
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
