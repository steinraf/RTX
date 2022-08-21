//
// Created by steinraf on 19/08/22.
//

#include "scene.h"

__host__ Scene::Scene(int width, int height) :  width(width), height(height),
                                                imageBufferSize(width*height*sizeof(Vector3f)),
                                                blockSize(width / blockSizeX + 1, height / blockSizeY + 1){

    checkCudaErrors(cudaMalloc((void **) &deviceImageBuffer, imageBufferSize));


    checkCudaErrors(cudaMalloc((void **) &deviceCurandState, width * height * sizeof(curandState)));



}

__host__ Scene::~Scene(){
//        checkCudaErrors(cudaDeviceSynchronize());
//        checkCudaErrors(cudaFree(deviceImageBuffer));

    cuda_helpers::freeVariables<<<blockSize, threadSize>>>(width, height);
}

void Scene::render() const{

    cuda_helpers::initRng<<<blockSize, threadSize>>>(width, height, deviceCurandState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cuda_helpers::initVariables<<<1, 1>>>(width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cuda_helpers::render<<<blockSize, threadSize>>>(deviceImageBuffer, width, height, deviceCurandState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());



    const std::string base_path = std::filesystem::path(__FILE__).parent_path().parent_path();
    const std::string path = base_path + "/data/image.ppm";
    const std::string png_path = base_path + "/data/image.png";

    auto hostImageBuffer = new Vector3f[imageBufferSize];

    checkCudaErrors(cudaMemcpy(hostImageBuffer, deviceImageBuffer, imageBufferSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    pngwriter png(width, height, 1., png_path.c_str());

    for (int j = height-1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            png.plot(i+1, j+1, hostImageBuffer[j * width + i][0], hostImageBuffer[j * width + i][1], hostImageBuffer[j * width + i][2]);
        }
    }

    delete[] hostImageBuffer;

    png.close();
}
