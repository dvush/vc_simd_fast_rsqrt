cmake_minimum_required(VERSION 3.1)
project(vector_norm)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-march=native -O3")

find_package(Vc REQUIRED)
include_directories(${Vc_INCLUDE_DIR})

link_directories(${Vc_LIB_DIR})
add_executable(vector_norm main.cpp)
target_link_libraries(vector_norm Vc)
