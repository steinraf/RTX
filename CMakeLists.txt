cmake_minimum_required(VERSION 3.20)
project(Raytracing LANGUAGES CXX CUDA)

set(CMAKE_CUDA_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-O3 -funroll-loops -DNO_FREETYPE -D_GLIBCXX_ASSERTIONS -fasynchronous-unwind-tables -fexceptions -fpie -Wl,-pie -fstack-clash-protection -fstack-protector-all -fstack-protector-strong -g -grecord-gcc-switches -fcf-protection -pipe -Wall -Werror=format-security  -Werror ")

find_package(PNGwriter 0.7.0 REQUIRED)


add_executable(Raytracing main.cu src/Camera.h src/Hittable.h src/hitable_list.h src/material.h src/Ray.h src/sphere.h src/Vector.h src/sphere.cu src/scenes.cu src/scenes.h src/util.h)

set_target_properties(Raytracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(Raytracing PRIVATE PNGwriter::PNGwriter)