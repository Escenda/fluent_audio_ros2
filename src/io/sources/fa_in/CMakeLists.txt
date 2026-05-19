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
  src/backends/factory.cpp
  src/backends/network_pcm_receiver_backend.cpp
  src/backends/pcm_file_reader_backend.cpp
)

target_include_directories(fa_in_backends
  PUBLIC
  include
  PRIVATE
  ${ALSA_INCLUDE_DIRS}
)
target_link_libraries(fa_in_backends PUBLIC ${ALSA_LIBRARIES})

add_library(fa_in_node_core
  src/fa_in_node.cpp
)

ament_target_dependencies(fa_in_node_core
  rclcpp
  std_srvs
  diagnostic_msgs
  fa_interfaces
)

target_include_directories(fa_in_node_core PUBLIC
  include
)
target_link_libraries(fa_in_node_core fa_in_backends)

add_executable(fa_in_node
  src/main.cpp
)

ament_target_dependencies(fa_in_node
  rclcpp
)
target_link_libraries(fa_in_node fa_in_node_core)

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
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  # Keep package tests isolated from ambient pytest plugins in ROS images.
  set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "1")
  ament_add_pytest_test(${PROJECT_NAME}_pytest test
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30
  )

  ament_add_gtest(${PROJECT_NAME}_audio_config_validation_test
    test/cpp/test_audio_config_validation.cpp
  )
  target_include_directories(${PROJECT_NAME}_audio_config_validation_test PRIVATE
    include
  )

  ament_add_gtest(${PROJECT_NAME}_source_backend_factory_test
    test/cpp/test_source_backend_factory.cpp
  )
  target_include_directories(${PROJECT_NAME}_source_backend_factory_test PRIVATE
    include
  )
  target_link_libraries(${PROJECT_NAME}_source_backend_factory_test
    fa_in_backends
  )

  ament_add_gtest(${PROJECT_NAME}_node_contract_test
    test/cpp/test_fa_in_node_contract.cpp
  )
  target_include_directories(${PROJECT_NAME}_node_contract_test PRIVATE
    include
  )
  ament_target_dependencies(${PROJECT_NAME}_node_contract_test
    rclcpp
    diagnostic_msgs
    fa_interfaces
  )
  target_link_libraries(${PROJECT_NAME}_node_contract_test
    fa_in_node_core
  )

  ament_add_gtest(${PROJECT_NAME}_network_pcm_receiver_backend_test
    test/cpp/test_network_pcm_receiver_backend.cpp
  )
  target_include_directories(${PROJECT_NAME}_network_pcm_receiver_backend_test PRIVATE
    include
  )
  target_link_libraries(${PROJECT_NAME}_network_pcm_receiver_backend_test
    fa_in_backends
  )

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
