cmake_minimum_required(VERSION 3.8)
project(fa_ns)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(fa_interfaces REQUIRED)

add_executable(fa_ns_node
  src/fa_ns_node.cpp
)

target_include_directories(fa_ns_node PRIVATE
  include
)

ament_target_dependencies(fa_ns_node
  ament_index_cpp
  rclcpp
  diagnostic_msgs
  fa_interfaces
)

option(FA_NS_ENABLE_ONNXRUNTIME "Enable ONNX Runtime backend" ON)

if(FA_NS_ENABLE_ONNXRUNTIME)
  set(ONNXRUNTIME_INCLUDE_DIR "" CACHE PATH "ONNX Runtime include directory (contains onnxruntime/core/session/onnxruntime_cxx_api.h)")
  set(ONNXRUNTIME_LIBRARY "" CACHE FILEPATH "ONNX Runtime library file (e.g. libonnxruntime.so)")

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
    message(STATUS "ONNX Runtime found: include=${ONNXRUNTIME_INCLUDE_DIR} lib=${ONNXRUNTIME_LIBRARY}")
    target_compile_definitions(fa_ns_node PRIVATE FA_NS_WITH_ONNXRUNTIME=1)
    target_include_directories(fa_ns_node PRIVATE
      ${ONNXRUNTIME_INCLUDE_DIR}
      third_party/kissfft
    )
    target_link_libraries(fa_ns_node ${ONNXRUNTIME_LIBRARY})
    target_sources(fa_ns_node PRIVATE
      src/dtln_onnx_engine.cpp
      third_party/kissfft/kiss_fft.c
      third_party/kissfft/kiss_fftr.c
    )
  else()
    message(WARNING "ONNX Runtime not found; building fa_ns without dtln_onnx backend. Set ONNXRUNTIME_ROOT or ONNXRUNTIME_INCLUDE_DIR/ONNXRUNTIME_LIBRARY.")
  endif()
endif()

install(TARGETS fa_ns_node
  DESTINATION lib/${PROJECT_NAME}
)

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
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
