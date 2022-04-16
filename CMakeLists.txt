cmake_minimum_required(VERSION 3.21)
project(Raytracing)

set(CMAKE_CXX_STANDARD 20)

add_executable(Raytracing main.cpp src/scene.cpp src/scene.h src/util.cpp src/util.h src/shapes.cpp src/shapes.h)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Raytracing OpenMP::OpenMP_CXX)
endif()