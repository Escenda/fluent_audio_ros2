cmake_minimum_required(VERSION 3.8)
project(fa_kws)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(fa_interfaces REQUIRED)

include_directories(include)

option(
  FA_KWS_ALLOW_TEST_ONLY_WITHOUT_SHERPA
  "Allow configuring fa_kws contract tests without building sherpa-onnx runtime targets"
  OFF
)

set(SHERPA_ONNX_PREFIX "" CACHE PATH "sherpa-onnx install prefix")
if(NOT SHERPA_ONNX_PREFIX AND DEFINED ENV{SHERPA_ONNX_PREFIX})
  set(SHERPA_ONNX_PREFIX "$ENV{SHERPA_ONNX_PREFIX}" CACHE PATH "sherpa-onnx install prefix" FORCE)
endif()

find_path(SHERPA_ONNX_INCLUDE_DIR
  NAMES sherpa-onnx/c-api/c-api.h
  HINTS
    "${SHERPA_ONNX_PREFIX}/include"
    /usr/local/include
    /usr/include
)

find_library(SHERPA_ONNX_C_API_LIBRARY
  NAMES sherpa-onnx-c-api
  HINTS
    "${SHERPA_ONNX_PREFIX}/lib"
    /usr/local/lib
    /usr/lib
)

set(FA_KWS_HAS_SHERPA_ONNX FALSE)
if(SHERPA_ONNX_INCLUDE_DIR AND SHERPA_ONNX_C_API_LIBRARY)
  set(FA_KWS_HAS_SHERPA_ONNX TRUE)
endif()

if(NOT FA_KWS_HAS_SHERPA_ONNX)
  if(BUILD_TESTING AND FA_KWS_ALLOW_TEST_ONLY_WITHOUT_SHERPA)
    message(WARNING
      "sherpa-onnx C API not found; configuring fa_kws contract tests only. "
      "Runtime targets fa_kws_node/fa_kws_wav_tool will not be built.")
  else()
    message(FATAL_ERROR
      "sherpa-onnx C API not found. Set SHERPA_ONNX_PREFIX or install sherpa-onnx "
      "before building fa_kws.")
  endif()
endif()

if(FA_KWS_HAS_SHERPA_ONNX)
  add_executable(fa_kws_node
    src/fa_kws_node.cpp
    src/audio_utils.cpp
    src/backends/sherpa_onnx_kws_backend.cpp
  )

  add_executable(fa_kws_wav_tool
    src/kws_wav_tool.cpp
    src/backends/sherpa_onnx_kws_backend.cpp
  )

  ament_target_dependencies(fa_kws_node
    rclcpp
    fa_interfaces
  )

  target_include_directories(fa_kws_node PRIVATE
    ${SHERPA_ONNX_INCLUDE_DIR}
  )

  target_link_libraries(fa_kws_node
    ${SHERPA_ONNX_C_API_LIBRARY}
  )

  get_filename_component(SHERPA_ONNX_LIB_DIR "${SHERPA_ONNX_C_API_LIBRARY}" DIRECTORY)
  set_target_properties(fa_kws_node PROPERTIES
    BUILD_RPATH "${SHERPA_ONNX_LIB_DIR}"
    INSTALL_RPATH "${SHERPA_ONNX_LIB_DIR}"
  )

  target_include_directories(fa_kws_wav_tool PRIVATE
    ${SHERPA_ONNX_INCLUDE_DIR}
  )

  target_link_libraries(fa_kws_wav_tool
    ${SHERPA_ONNX_C_API_LIBRARY}
  )

  set_target_properties(fa_kws_wav_tool PROPERTIES
    BUILD_RPATH "${SHERPA_ONNX_LIB_DIR}"
    INSTALL_RPATH "${SHERPA_ONNX_LIB_DIR}"
  )

  install(TARGETS fa_kws_node fa_kws_wav_tool
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  ament_add_gtest(${PROJECT_NAME}_vad_gate_test
    test/unit/vad_gate_contract.cpp
  )
  if(TARGET ${PROJECT_NAME}_vad_gate_test)
    target_include_directories(${PROJECT_NAME}_vad_gate_test PRIVATE
      include
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
