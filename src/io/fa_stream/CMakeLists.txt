cmake_minimum_required(VERSION 3.8)
project(fa_stream)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/radio_streamer.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

