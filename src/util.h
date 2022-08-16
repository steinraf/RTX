//
// Created by steinraf on 10/08/22.
//

#pragma once

#include <utility>
#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

namespace gpu{

    template<typename T>
    class iterator{

    };

    template<typename T, size_t size>
    class array{
        __device__ array(T data[size]) : data(data){}
        __device__ array(std::initializer_list<T> l):data(l){}

        iterator<T> begin(){return &data;}
        iterator<T> end(){return &data[size-1];}



        T data[size];
    };
}
