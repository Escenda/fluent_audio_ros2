cmake_minimum_required(VERSION 3.8)
project(fa_tts)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(fa_interfaces REQUIRED)

ament_python_install_package(fa_tts_py PACKAGE_DIR fa_tts_py)

install(PROGRAMS
  scripts/fa_tts_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY resource/
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(
  rclpy
  fa_interfaces
)

ament_package()
