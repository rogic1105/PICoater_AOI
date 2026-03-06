cmake_minimum_required(VERSION 3.20)

project(picoater_plc LANGUAGES CXX)

add_library(picoater_plc STATIC
  mock_plc_adapter.cpp
)

target_include_directories(picoater_plc
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(picoater_plc PUBLIC cxx_std_17)
