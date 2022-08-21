//
// Created by steinraf on 19/08/22.
//

#pragma once

#include <curand_kernel.h>

#include "vector.h"
#include "ray.h"
#include "hittableList.h"

#define checkCudaErrors(val) cuda_helpers::check_cuda( (val), #val, __FILE__, __LINE__ )


namespace cuda_helpers {

    __host__ void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);


    __global__ void initRng(int width, int height, curandState *randState);

    __global__ void initVariables(Hittable ** hittables, HittableList **hittableList, size_t numHittables, int width, int height);
    __global__ void freeVariables(int width, int height);

    __device__ Color getColor(const Ray& r);

    __device__ bool hitSphere(const Vector3f& center, float radius, const Ray&r);


    __global__ void render(Vector3f *output, HittableList **hittableList, int width, int height,  curandState *localRandState);


    __device__ bool inline initIndices(int &i, int &j, int &pixelIndex, const int width, const int height){
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = threadIdx.y + blockIdx.y * blockDim.y;

        if ((i >= width) || (j >= height)) return false;

        pixelIndex = j * width + i;

        return true;
    }

};


