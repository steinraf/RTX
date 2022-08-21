//
// Created by steinraf on 19/08/22.
//

#pragma once

#include <filesystem>

#include <pngwriter.h>

#include "cuda_helpers.h"
#include "vector.h"
#include "hittableList.h"




class Scene {
public:
    __host__ Scene() = delete;
    __host__ Scene(int width=384, int height=216);

    __host__ ~Scene();

    void render() const;

private:

    const unsigned int blockSizeX = 8, blockSizeY = 8;

    const dim3 threadSize{blockSizeX, blockSizeY};

    const dim3 blockSize;

    Vector3f *deviceImageBuffer;
    const size_t imageBufferSize;

    Hittable **deviceHittables;
    const size_t numHittables = 2;

    HittableList **deviceHittableList;

    curandState *deviceCurandState;




    const int width, height;

};

