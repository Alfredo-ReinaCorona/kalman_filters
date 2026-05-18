cmake_minimum_required(VERSION 3.16)

project(kalman_filter_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

add_executable(kalman kalman_filter_4d_2d.cpp)

target_link_libraries(kalman PRIVATE Eigen3::Eigen)

target_compile_options(kalman PRIVATE -Wall -Wextra -Wpedantic)
