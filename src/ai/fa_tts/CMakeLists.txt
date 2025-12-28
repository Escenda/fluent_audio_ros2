cmake_minimum_required(VERSION 3.8)
project(fv_tts)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(fv_audio REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(service_files
  "srv/Speak.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${service_files}
  DEPENDENCIES fv_audio
)

ament_python_install_package(fv_tts_py PACKAGE_DIR fv_tts_py)

install(PROGRAMS
  scripts/fv_tts_node
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
  fv_audio
  rosidl_default_runtime
)

ament_package()
