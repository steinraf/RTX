//
// Created by steinraf on 19/08/22.
//

#include "cuda_helpers.h"
#include "ray.h"

#include <iostream>



namespace cuda_helpers{

    __host__ void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
        if (result) {
            std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                      file << ":" << line << " '" << func << "' \n";
            cudaDeviceReset();
            exit(99);
        }
    }




    __global__ void initRng(int width, int height, curandState *randState){
        int i, j, pixelIndex;
        if (!initIndices(i, j, pixelIndex, width, height)) return;

        curand_init(42, pixelIndex, 0, &randState[pixelIndex]);
    }

    __global__ void initVariables(int width, int height){
        int i, j, pixelIndex;
        if (!initIndices(i, j, pixelIndex, 1, 1)) return;

        Vector3f origin{0, 0, 0};
        Vector3f lookAt(0, 0, -1);
        const float distToFocus = (lookAt - origin).norm();

    }

    __global__ void freeVariables(int width, int height){
        int i, j, pixelIndex;
        if (!initIndices(i, j, pixelIndex, 1, 1)) return;


    }

    __device__ Color getColor(const Ray& r){

        if(hitSphere({0,0,-1}, 0.5, r))
            return {1.0f, 0, 0};

        auto t = 0.5*(r.getDirection().normalized()[1] + 1.f);

        return (1-t)*Vector3f{1.f} + t*Color{0.5f, 0.7f, 1.0f};
    }

    __device__ bool hitSphere(const Vector3f& center, float radius, const Ray&r){
        auto oc = r.getOrigin() - center;
        const float a = r.getDirection().squaredNorm();
        const float b = 2.0f * oc.dot(r.getDirection());
        const float c = oc.squaredNorm() - radius*radius;
        const float disc = b*b - 4*a*c;
        return disc > 0;
    }


    __global__ void render(Vector3f *output, int width, int height, curandState *localRandState){
        int i, j, pixelIndex;
        if (!initIndices(i, j, pixelIndex, width, height)) return;

//        const float u = (i + curand_uniform(localRandState)) / width;
//        const float v = (j + curand_uniform(localRandState)) / height;

        const float u = static_cast<float>(i)/(width-1);
        const float v = static_cast<float>(j)/(height-1);


        const float aspectRatio = static_cast<float>(width)/height;

        const float viewportHeight = 2.0f;
        const float viewportWidth = aspectRatio * viewportHeight;
        const float focalLength = 1.0f;

        Vector3f    origin{0.0f},
                    horizontal{viewportWidth, 0, 0},
                    vertical{0, viewportHeight, 0},
                    lowerLeftCorner = origin-horizontal/2.0f-vertical/2.0f - Vector3f{0, 0, focalLength};

        Ray ray{origin, lowerLeftCorner + u*horizontal + v*vertical - origin};

        auto col = getColor(ray);

        output[pixelIndex] = col;
    }



}
