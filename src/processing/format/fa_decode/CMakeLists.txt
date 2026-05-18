cmake_minimum_required(VERSION 3.8)
project(fa_decode)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(fa_interfaces REQUIRED)

add_library(fa_decode_external_codec_decoder
  src/backends/external_codec_decoder.cpp
)

target_include_directories(fa_decode_external_codec_decoder PUBLIC
  include
)

add_library(fa_decode_node_core
  src/fa_decode_node.cpp
)

target_include_directories(fa_decode_node_core PUBLIC
  include
)

ament_target_dependencies(fa_decode_node_core
  rclcpp
  diagnostic_msgs
  fa_interfaces
)

target_link_libraries(fa_decode_node_core
  fa_decode_external_codec_decoder
)

add_executable(fa_decode_node
  src/main.cpp
)

target_link_libraries(fa_decode_node
  fa_decode_node_core
)

install(TARGETS fa_decode_node
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

  ament_add_gtest(${PROJECT_NAME}_backend_test
    test/cpp/test_external_codec_decoder_backend.cpp
  )
  target_include_directories(${PROJECT_NAME}_backend_test PRIVATE
    include
  )
  target_link_libraries(${PROJECT_NAME}_backend_test
    fa_decode_external_codec_decoder
  )

  ament_add_gtest(${PROJECT_NAME}_graph_smoke_test
    test/cpp/test_fa_decode_graph.cpp
  )
  target_include_directories(${PROJECT_NAME}_graph_smoke_test PRIVATE
    include
  )
  target_link_libraries(${PROJECT_NAME}_graph_smoke_test
    fa_decode_node_core
  )
  ament_target_dependencies(${PROJECT_NAME}_graph_smoke_test
    rclcpp
    fa_interfaces
  )

  set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "1")
  ament_add_pytest_test(${PROJECT_NAME}_pytest test
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30
  )
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
