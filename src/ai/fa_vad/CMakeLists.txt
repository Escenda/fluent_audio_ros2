cmake_minimum_required(VERSION 3.8)
project(fv_audio_vad)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(fv_audio REQUIRED)

include_directories(include)

add_executable(fv_audio_vad_node
  src/fv_audio_vad_node.cpp
)

ament_target_dependencies(fv_audio_vad_node
  rclcpp
  std_msgs
  fv_audio
)

install(TARGETS fv_audio_vad_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# Only install include directory if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
  install(DIRECTORY include/
    DESTINATION include/
  )
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
