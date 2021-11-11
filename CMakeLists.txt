cmake_minimum_required(VERSION 3.19)
project(lgtm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fopenmp -O0")

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(make-aknng make-aknng.cpp)
add_executable(make-index make-index.cpp)
