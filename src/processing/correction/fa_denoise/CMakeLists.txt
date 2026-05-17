cmake_minimum_required(VERSION 3.8)
project(fa_denoise)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(fa_interfaces REQUIRED)

set(FA_DENOISE_ONNXRUNTIME "AUTO" CACHE STRING "ONNX Runtime backend build mode: AUTO, ON, or OFF")
set_property(CACHE FA_DENOISE_ONNXRUNTIME PROPERTY STRINGS AUTO ON OFF)

if(NOT FA_DENOISE_ONNXRUNTIME MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "FA_DENOISE_ONNXRUNTIME must be AUTO, ON, or OFF")
endif()

set(ONNXRUNTIME_INCLUDE_DIR "" CACHE PATH "ONNX Runtime include directory (contains onnxruntime/core/session/onnxruntime_cxx_api.h)")
set(ONNXRUNTIME_LIBRARY "" CACHE FILEPATH "ONNX Runtime library file (e.g. libonnxruntime.so)")

set(FA_DENOISE_WITH_ONNXRUNTIME OFF)

if(NOT FA_DENOISE_ONNXRUNTIME STREQUAL "OFF")
  if(NOT ONNXRUNTIME_INCLUDE_DIR)
    find_path(ONNXRUNTIME_INCLUDE_DIR
      NAMES onnxruntime/core/session/onnxruntime_cxx_api.h
      HINTS ENV ONNXRUNTIME_ROOT ENV ONNXRUNTIME_DIR
      PATH_SUFFIXES include
    )
  endif()

  if(NOT ONNXRUNTIME_LIBRARY)
    find_library(ONNXRUNTIME_LIBRARY
      NAMES onnxruntime
      HINTS ENV ONNXRUNTIME_ROOT ENV ONNXRUNTIME_DIR
      PATH_SUFFIXES lib lib64
    )
  endif()

  if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    set(FA_DENOISE_WITH_ONNXRUNTIME ON)
  elseif(FA_DENOISE_ONNXRUNTIME STREQUAL "ON")
    message(FATAL_ERROR "ONNX Runtime is required when FA_DENOISE_ONNXRUNTIME=ON. Set ONNXRUNTIME_ROOT or ONNXRUNTIME_INCLUDE_DIR/ONNXRUNTIME_LIBRARY.")
  else()
    message(STATUS "ONNX Runtime not found; building fa_denoise without dtln_onnx runtime. Selecting backend.name=dtln_onnx will fail closed at node startup.")
  endif()
endif()

if(FA_DENOISE_WITH_ONNXRUNTIME)
  message(STATUS "ONNX Runtime found: include=${ONNXRUNTIME_INCLUDE_DIR} lib=${ONNXRUNTIME_LIBRARY}")
  add_library(fa_denoise_backends STATIC
    src/backends/dtln_onnx_engine.cpp
    third_party/kissfft/kiss_fft.c
    third_party/kissfft/kiss_fftr.c
  )

  target_include_directories(fa_denoise_backends PUBLIC
    include
  )

  target_include_directories(fa_denoise_backends PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    third_party/kissfft
  )

  target_link_libraries(fa_denoise_backends PUBLIC
    ${ONNXRUNTIME_LIBRARY}
  )
endif()

add_executable(fa_denoise_node
  src/fa_denoise_node.cpp
)

target_include_directories(fa_denoise_node PRIVATE
  include
)

ament_target_dependencies(fa_denoise_node
  rclcpp
  diagnostic_msgs
  fa_interfaces
)

if(FA_DENOISE_WITH_ONNXRUNTIME)
  target_compile_definitions(fa_denoise_node PRIVATE FA_DENOISE_WITH_ONNXRUNTIME=1)

  target_link_libraries(fa_denoise_node
    fa_denoise_backends
  )

  get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
  set_target_properties(fa_denoise_node PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
  )
endif()

install(TARGETS fa_denoise_node
  DESTINATION lib/${PROJECT_NAME}
)

if(FA_DENOISE_WITH_ONNXRUNTIME)
  install(TARGETS fa_denoise_backends
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

install(DIRECTORY
  launch
  config
  models
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  third_party/dtln/LICENSE
  DESTINATION share/${PROJECT_NAME}/third_party/dtln
)

install(FILES
  third_party/kissfft/COPYING
  third_party/kissfft/LICENSE_BSD-3-Clause
  DESTINATION share/${PROJECT_NAME}/third_party/kissfft
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
