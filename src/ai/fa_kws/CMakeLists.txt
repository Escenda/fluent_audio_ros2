cmake_minimum_required(VERSION 3.8)
project(fa_kws)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(fa_interfaces REQUIRED)

include_directories(include)

set(FA_KWS_SHERPA_ONNX "AUTO" CACHE STRING "sherpa-onnx backend build mode: AUTO, ON, or OFF")
set_property(CACHE FA_KWS_SHERPA_ONNX PROPERTY STRINGS AUTO ON OFF)

if(NOT FA_KWS_SHERPA_ONNX MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "FA_KWS_SHERPA_ONNX must be AUTO, ON, or OFF")
endif()

set(SHERPA_ONNX_PREFIX "" CACHE PATH "sherpa-onnx install prefix")
if(NOT SHERPA_ONNX_PREFIX AND DEFINED ENV{SHERPA_ONNX_PREFIX})
  set(SHERPA_ONNX_PREFIX "$ENV{SHERPA_ONNX_PREFIX}" CACHE PATH "sherpa-onnx install prefix" FORCE)
endif()

set(FA_KWS_WITH_SHERPA_ONNX OFF)

if(NOT FA_KWS_SHERPA_ONNX STREQUAL "OFF")
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

  if(SHERPA_ONNX_INCLUDE_DIR AND SHERPA_ONNX_C_API_LIBRARY)
    set(FA_KWS_WITH_SHERPA_ONNX ON)
  elseif(FA_KWS_SHERPA_ONNX STREQUAL "ON")
    message(FATAL_ERROR
      "sherpa-onnx C API is required when FA_KWS_SHERPA_ONNX=ON. "
      "Set SHERPA_ONNX_PREFIX or install sherpa-onnx before building fa_kws.")
  else()
    message(STATUS
      "sherpa-onnx C API not found; building fa_kws_node without sherpa_onnx_kws runtime. "
      "Selecting backend.name=sherpa_onnx_kws will fail closed at node startup.")
  endif()
endif()

add_executable(fa_kws_node
  src/fa_kws_node.cpp
  src/audio_utils.cpp
)

ament_target_dependencies(fa_kws_node
  rclcpp
  fa_interfaces
)

if(FA_KWS_WITH_SHERPA_ONNX)
  target_compile_definitions(fa_kws_node PRIVATE FA_KWS_WITH_SHERPA_ONNX=1)
endif()

if(FA_KWS_WITH_SHERPA_ONNX)
  add_library(fa_kws_backends STATIC
    src/backends/sherpa_onnx_kws_backend.cpp
  )

  target_include_directories(fa_kws_backends PUBLIC
    include
    ${SHERPA_ONNX_INCLUDE_DIR}
  )

  target_link_libraries(fa_kws_backends PUBLIC
    ${SHERPA_ONNX_C_API_LIBRARY}
  )

  add_executable(fa_kws_wav_tool
    src/kws_wav_tool.cpp
  )

  target_link_libraries(fa_kws_node
    fa_kws_backends
  )

  get_filename_component(SHERPA_ONNX_LIB_DIR "${SHERPA_ONNX_C_API_LIBRARY}" DIRECTORY)
  set_target_properties(fa_kws_node PROPERTIES
    BUILD_RPATH "${SHERPA_ONNX_LIB_DIR}"
    INSTALL_RPATH "${SHERPA_ONNX_LIB_DIR}"
  )

  target_link_libraries(fa_kws_wav_tool
    fa_kws_backends
  )

  set_target_properties(fa_kws_wav_tool PROPERTIES
    BUILD_RPATH "${SHERPA_ONNX_LIB_DIR}"
    INSTALL_RPATH "${SHERPA_ONNX_LIB_DIR}"
  )

  install(TARGETS fa_kws_node fa_kws_wav_tool
    DESTINATION lib/${PROJECT_NAME}
  )
else()
  install(TARGETS fa_kws_node
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
