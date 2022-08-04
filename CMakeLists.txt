cmake_minimum_required(VERSION 3.20)
project(GPU_Raytracing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-O3 -funroll-loops -D_GLIBCXX_ASSERTIONS -fopenmp -fasynchronous-unwind-tables -fexceptions -fpie -Wl,-pie -fstack-clash-protection -fstack-protector-all -fstack-protector-strong -g -grecord-gcc-switches -fcf-protection -pipe -Wall -Werror=format-security  -Werror ")

add_executable(GPU_Raytracing main.cu src/scene.cpp src/scene.h)