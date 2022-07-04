cmake_minimum_required(VERSION 3.10)
project(vpic-reader C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)
find_package(Matplot++ REQUIRED)

add_executable(pp post_processor.cc)
target_link_libraries(pp Threads::Threads)
add_executable(qr query_runner.cc)
target_link_libraries(qr Matplot++::matplot)
