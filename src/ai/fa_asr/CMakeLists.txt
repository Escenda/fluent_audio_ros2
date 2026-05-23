cmake_minimum_required(VERSION 3.8)
project(fa_asr)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(fa_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

ament_python_install_package(fa_asr_py PACKAGE_DIR fa_asr_py)

install(PROGRAMS
  scripts/fa_asr_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  models
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY resource/
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(
  rclpy
  fa_interfaces
  std_msgs
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)

  set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "1")
  ament_add_pytest_test(${PROJECT_NAME}_pytest test
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60
  )
endif()

ament_package()
