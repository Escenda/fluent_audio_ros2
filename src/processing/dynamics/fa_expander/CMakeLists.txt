cmake_minimum_required(VERSION 3.8)
project(fa_expander)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(fa_interfaces REQUIRED)

add_library(fa_expander_internal_static_expander STATIC
  src/backends/internal_static_expander.cpp
)

target_include_directories(fa_expander_internal_static_expander PUBLIC
  include
)

add_executable(fa_expander_node
  src/fa_expander_node.cpp
)

target_include_directories(fa_expander_node PRIVATE
  include
)

ament_target_dependencies(fa_expander_node
  rclcpp
  diagnostic_msgs
  fa_interfaces
)

target_link_libraries(fa_expander_node
  fa_expander_internal_static_expander
)

install(TARGETS fa_expander_node
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
    test/cpp/test_internal_static_expander_backend.cpp
  )
  if(TARGET ${PROJECT_NAME}_backend_test)
    target_include_directories(${PROJECT_NAME}_backend_test PRIVATE
      include
    )
    target_link_libraries(${PROJECT_NAME}_backend_test
      fa_expander_internal_static_expander
    )
  endif()

  set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "1")
  ament_add_pytest_test(${PROJECT_NAME}_pytest test
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30
  )
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
