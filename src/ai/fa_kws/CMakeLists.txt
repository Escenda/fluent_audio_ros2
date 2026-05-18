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

set(FA_KWS_SHERPA_ONNX "OFF" CACHE STRING "sherpa-onnx backend build mode: ON or OFF")
set_property(CACHE FA_KWS_SHERPA_ONNX PROPERTY STRINGS ON OFF)

if(NOT FA_KWS_SHERPA_ONNX MATCHES "^(ON|OFF)$")
  message(FATAL_ERROR "FA_KWS_SHERPA_ONNX must be ON or OFF")
endif()

set(SHERPA_ONNX_PREFIX "" CACHE PATH "sherpa-onnx install prefix")
if(NOT SHERPA_ONNX_PREFIX AND DEFINED ENV{SHERPA_ONNX_PREFIX})
  set(SHERPA_ONNX_PREFIX "$ENV{SHERPA_ONNX_PREFIX}" CACHE PATH "sherpa-onnx install prefix" FORCE)
endif()

set(FA_KWS_WITH_SHERPA_ONNX OFF)
set(FA_KWS_BACKEND_SOURCE src/backends/sherpa_onnx_kws_backend_unavailable.cpp)

if(FA_KWS_SHERPA_ONNX STREQUAL "ON")
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
    set(FA_KWS_BACKEND_SOURCE src/backends/sherpa_onnx_kws_backend.cpp)
  else()
    message(FATAL_ERROR
      "sherpa-onnx C API is required when FA_KWS_SHERPA_ONNX=ON. "
      "Set SHERPA_ONNX_PREFIX or install sherpa-onnx before building fa_kws.")
  endif()
else()
  message(STATUS
    "FA_KWS_SHERPA_ONNX=OFF builds fa_kws runtime targets with a fail-closed "
    "unavailable sherpa_onnx_kws backend.")
endif()

add_library(fa_kws_backends STATIC
  ${FA_KWS_BACKEND_SOURCE}
)

target_include_directories(fa_kws_backends PUBLIC
  include
)

if(FA_KWS_WITH_SHERPA_ONNX)
  target_include_directories(fa_kws_backends PUBLIC
    ${SHERPA_ONNX_INCLUDE_DIR}
  )

  target_link_libraries(fa_kws_backends PUBLIC
    ${SHERPA_ONNX_C_API_LIBRARY}
  )
endif()

add_executable(fa_kws_wav_tool
  src/kws_wav_tool.cpp
)

add_executable(fa_kws_node
  src/fa_kws_node.cpp
  src/audio_utils.cpp
  src/vad_state_identity.cpp
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

if(FA_KWS_WITH_SHERPA_ONNX)
  get_filename_component(SHERPA_ONNX_LIB_DIR "${SHERPA_ONNX_C_API_LIBRARY}" DIRECTORY)
  set_target_properties(fa_kws_node PROPERTIES
    BUILD_RPATH "${SHERPA_ONNX_LIB_DIR}"
    INSTALL_RPATH "${SHERPA_ONNX_LIB_DIR}"
  )

  set_target_properties(fa_kws_wav_tool PROPERTIES
    BUILD_RPATH "${SHERPA_ONNX_LIB_DIR}"
    INSTALL_RPATH "${SHERPA_ONNX_LIB_DIR}"
  )
endif()

install(TARGETS fa_kws_node fa_kws_wav_tool
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

  ament_add_gtest(${PROJECT_NAME}_vad_gate_test
    test/unit/vad_gate_contract.cpp
  )
  if(TARGET ${PROJECT_NAME}_vad_gate_test)
    target_include_directories(${PROJECT_NAME}_vad_gate_test PRIVATE
      include
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

  ament_add_gtest(${PROJECT_NAME}_vad_state_identity_test
    test/unit/vad_state_identity_contract.cpp
    src/vad_state_identity.cpp
  )
  if(TARGET ${PROJECT_NAME}_vad_state_identity_test)
    target_include_directories(${PROJECT_NAME}_vad_state_identity_test PRIVATE
      include
    )
    ament_target_dependencies(${PROJECT_NAME}_vad_state_identity_test
      fa_interfaces
    )
  endif()

  if(NOT FA_KWS_WITH_SHERPA_ONNX)
    ament_add_gtest(${PROJECT_NAME}_sherpa_unavailable_backend_test
      test/unit/sherpa_unavailable_backend_contract.cpp
    )
    if(TARGET ${PROJECT_NAME}_sherpa_unavailable_backend_test)
      target_include_directories(${PROJECT_NAME}_sherpa_unavailable_backend_test PRIVATE
        include
      )
      target_link_libraries(${PROJECT_NAME}_sherpa_unavailable_backend_test
        fa_kws_backends
      )
    endif()
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
