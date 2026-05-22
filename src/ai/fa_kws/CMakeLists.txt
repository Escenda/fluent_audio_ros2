cmake_minimum_required(VERSION 3.8)
project(fa_kws)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(fa_interfaces REQUIRED)

include_directories(include)

add_library(fa_kws_backends STATIC
  src/backends/factory.cpp
  src/backends/sherpa_onnx_kws_backend.cpp
)

target_include_directories(fa_kws_backends PUBLIC
  include
)

add_executable(fa_kws_wav_tool
  src/kws_wav_tool.cpp
)

add_executable(fa_kws_node
  src/fa_kws_node.cpp
  src/audio_utils.cpp
)

ament_target_dependencies(fa_kws_node
  rclcpp
  fa_interfaces
)

target_link_libraries(fa_kws_node
  fa_kws_backends
)

target_link_libraries(fa_kws_wav_tool
  fa_kws_backends
)

install(TARGETS fa_kws_node fa_kws_wav_tool
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/sherpa_onnx_kws_worker
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  fa_kws_py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  ament_add_gtest(${PROJECT_NAME}_backend_config_validation_test
    test/unit/backend_config_validation_contract.cpp
  )
  if(TARGET ${PROJECT_NAME}_backend_config_validation_test)
    target_include_directories(${PROJECT_NAME}_backend_config_validation_test PRIVATE
      include
    )
  endif()

  ament_add_gtest(${PROJECT_NAME}_backend_factory_test
    test/unit/backend_factory_contract.cpp
  )
  if(TARGET ${PROJECT_NAME}_backend_factory_test)
    target_include_directories(${PROJECT_NAME}_backend_factory_test PRIVATE
      include
    )
    target_link_libraries(${PROJECT_NAME}_backend_factory_test
      fa_kws_backends
    )
  endif()

  ament_add_gtest(${PROJECT_NAME}_audio_utils_test
    test/unit/audio_utils_contract.cpp
    src/audio_utils.cpp
  )
  if(TARGET ${PROJECT_NAME}_audio_utils_test)
    target_include_directories(${PROJECT_NAME}_audio_utils_test PRIVATE
      include
    )
    ament_target_dependencies(${PROJECT_NAME}_audio_utils_test
      fa_interfaces
    )
  endif()

  # Keep package tests isolated from ambient pytest plugins in ROS images.
  set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "1")
  ament_add_pytest_test(${PROJECT_NAME}_pytest test
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30
  )

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
