cmake_minimum_required(VERSION 3.20)
project(Raytracing)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-O3 -D_GLIBCXX_ASSERTIONS -fopenmp -fasynchronous-unwind-tables -fexceptions -fpie -Wl,-pie -fstack-clash-protection -fstack-protector-all -fstack-protector-strong -g -grecord-gcc-switches -fcf-protection -pipe -Wall -Werror=format-security  -Werror ")

add_executable(Raytracing main.cpp src/scene.cpp src/scene.h src/util.cpp src/util.h src/shapes.cpp src/shapes.h src/materials.cpp src/materials.h)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(Raytracing OpenMP::OpenMP_CXX)
endif ()