cmake_minimum_required(VERSION 3.8)
project(fa_bit_depth)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(fa_interfaces REQUIRED)

add_library(fa_bit_depth_internal_integer_bit_depth
  src/backends/internal_integer_bit_depth.cpp
)

target_include_directories(fa_bit_depth_internal_integer_bit_depth PUBLIC
  include
)

add_library(fa_bit_depth_node_core
  src/fa_bit_depth_node.cpp
)

target_include_directories(fa_bit_depth_node_core PUBLIC
  include
)

ament_target_dependencies(fa_bit_depth_node_core
  rclcpp
  diagnostic_msgs
  fa_interfaces
)

target_link_libraries(fa_bit_depth_node_core
  fa_bit_depth_internal_integer_bit_depth
)

add_executable(fa_bit_depth_node
  src/main.cpp
)

target_link_libraries(fa_bit_depth_node fa_bit_depth_node_core)

ament_target_dependencies(fa_bit_depth_node
  rclcpp
)

install(TARGETS fa_bit_depth_node
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
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "1")
  ament_add_pytest_test(${PROJECT_NAME}_pytest test
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30
  )

  ament_add_gtest(${PROJECT_NAME}_node_contract_test
    test/cpp/test_fa_bit_depth_node_contract.cpp
  )
  target_include_directories(${PROJECT_NAME}_node_contract_test PRIVATE
    include
  )
  ament_target_dependencies(${PROJECT_NAME}_node_contract_test
    rclcpp
    fa_interfaces
  )
  target_link_libraries(${PROJECT_NAME}_node_contract_test
    fa_bit_depth_node_core
  )

  ament_add_gtest(${PROJECT_NAME}_backend_test
    test/cpp/test_internal_integer_bit_depth_backend.cpp
  )
  target_include_directories(${PROJECT_NAME}_backend_test PRIVATE
    include
  )
  target_link_libraries(${PROJECT_NAME}_backend_test
    fa_bit_depth_internal_integer_bit_depth
  )

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
