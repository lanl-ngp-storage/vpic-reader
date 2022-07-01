cmake_minimum_required(VERSION 3.10)
project(vpic-reader C CXX)
set(CMAKE_CXX_STANDARD 11)

add_executable(pp post_processor.cc)
