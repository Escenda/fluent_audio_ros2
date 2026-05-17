cmake_minimum_required(VERSION 3.8)
project(fa_record)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(fa_interfaces REQUIRED)

add_library(fa_record_backends STATIC
  src/backends/file_writer_backend.cpp
)

target_include_directories(fa_record_backends PUBLIC
  include
)

add_executable(fa_record_node
  src/fa_record_node.cpp
)

target_include_directories(fa_record_node PRIVATE
  include
)

ament_target_dependencies(fa_record_node
  rclcpp
  fa_interfaces
)

target_link_libraries(fa_record_node
  fa_record_backends
)

install(TARGETS
  fa_record_node
  fa_record_backends
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
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "1")
  ament_add_pytest_test(${PROJECT_NAME}_pytest test
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30
  )

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
