cmake_minimum_required(VERSION 3.14)
project(agbs)

set(CMAKE_CXX_STANDARD 17)
set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++17 -Wall -Wextra -O3 -fopenmp -fno-exceptions -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

add_executable(agbs main.cpp)