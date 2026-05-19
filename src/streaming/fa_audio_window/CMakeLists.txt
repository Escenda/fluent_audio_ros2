cmake_minimum_required(VERSION 3.8)
project(fa_audio_window)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(fa_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_library(fa_audio_window_core
  src/audio_window_buffer.cpp
  src/fa_audio_window_node.cpp
  src/time_range_parser.cpp
  src/wav_writer.cpp
)

target_include_directories(fa_audio_window_core PUBLIC
  include
)

ament_target_dependencies(fa_audio_window_core
  fa_interfaces
  rclcpp
  std_msgs
)

add_executable(fa_audio_window_node
  src/main.cpp
)

target_link_libraries(fa_audio_window_node fa_audio_window_core)

ament_target_dependencies(fa_audio_window_node
  rclcpp
)

install(TARGETS fa_audio_window_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  ament_add_gtest(${PROJECT_NAME}_buffer_test
    test/cpp/test_audio_window_buffer.cpp
  )
  target_include_directories(${PROJECT_NAME}_buffer_test PRIVATE include)
  target_link_libraries(${PROJECT_NAME}_buffer_test fa_audio_window_core)

  ament_add_gtest(${PROJECT_NAME}_wav_writer_test
    test/cpp/test_wav_writer.cpp
  )
  target_include_directories(${PROJECT_NAME}_wav_writer_test PRIVATE include)
  target_link_libraries(${PROJECT_NAME}_wav_writer_test fa_audio_window_core)

  ament_add_gtest(${PROJECT_NAME}_service_contract_test
    test/cpp/test_export_service_contract.cpp
  )
  target_include_directories(${PROJECT_NAME}_service_contract_test PRIVATE include)
  ament_target_dependencies(${PROJECT_NAME}_service_contract_test
    fa_interfaces
    rclcpp
  )
  target_link_libraries(${PROJECT_NAME}_service_contract_test fa_audio_window_core)

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
