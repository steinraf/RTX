cmake_minimum_required(VERSION 3.20)
project(Raytracing LANGUAGES CXX CUDA)

set(CMAKE_CUDA_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-O3 -funroll-loops -DNO_FREETYPE -D_GLIBCXX_ASSERTIONS -fasynchronous-unwind-tables -fexceptions -fpie -Wl,-pie -fstack-clash-protection -fstack-protector-all -fstack-protector-strong -g -grecord-gcc-switches -fcf-protection -pipe -Wall -Werror ")

find_package(PNGwriter 0.7.0 REQUIRED)


add_executable(Raytracing main.cu src/vector.h src/util.cu src/util.h src/cuda_helpers.cu src/cuda_helpers.h src/scene.cu src/scene.h src/ray.cu src/ray.h src/hittable.cu src/hittable.h src/sphere.cu src/sphere.h src/hittableList.cu src/hittableList.h)
set_target_properties(Raytracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(Raytracing PRIVATE PNGwriter::PNGwriter)