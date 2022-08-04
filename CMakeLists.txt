cmake_minimum_required(VERSION 3.20)
project(Raytracing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-O3 -funroll-loops -D_GLIBCXX_ASSERTIONS -fasynchronous-unwind-tables -fexceptions -fpie -Wl,-pie -fstack-clash-protection -fstack-protector-all -fstack-protector-strong -g -grecord-gcc-switches -fcf-protection -pipe -Wall -Werror=format-security  -Werror ")

add_executable(Raytracing main.cu src/Camera.h src/Hittable.h src/hitable_list.h src/material.h src/Ray.h src/Sphere.h src/Vector.h)

set_target_properties(Raytracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)