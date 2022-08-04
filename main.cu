#include <iostream>
#include <filesystem>
#include <fstream>
#include "src/util.h"


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit( result);
    }
}

__device__ Color getBackground(const Ray &ray) {

    double grad = (ray.dir[1] + 1) / 2; // Map from -1 1 to 0 1
    return (1 - grad) * Color{1.0, 1.0, 1.0} + grad * Color{0.5, 0.7, 1.0};
}

__device__ Color castRay(const Ray& r, unsigned int rayDepth){

    if(rayDepth == 0)
        return Color{};

    return getBackground(r);
}

__global__ void render(Color *fb, size_t width, size_t height) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= width) || (j >= height)) return;

    unsigned int subSamples = 1;
    unsigned int rayDepth = 1;
    Color c;

    for(int i = 0; i < subSamples; ++i){
//        auto ray = camera[0]->getRay(i + getRandom(), j + getRandom());
        Ray r{{float(i)/width, float(j)/width, 0.0f},{0.0f, 0.0f, -1.0f}};
        c += castRay(r, rayDepth);
    }

    int pixel_index = j*width + i;
    fb[pixel_index] = c;
}

int main() {

    size_t width = 384, height = 216;
    size_t block_x = 8, block_y = 8;

    Color *pixels;
    checkCudaErrors(cudaMallocManaged((void **)&pixels, width*height*sizeof(Color)));

    std::cout << "Starting rendering...\n";

    dim3 blocks(width/block_x+1,height/block_y+1);
    dim3 threads(block_x,block_y);
    render<<<blocks, threads>>>(pixels, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    const std::string current_path = std::filesystem::path(__FILE__).parent_path();
    const std::string path = current_path + "/data/image.ppm";

    std::ofstream file{path};

    std::cout << "Ended rendering, writing to file...\n";

    // Output FB as Image
    file << "P3\n" << width << " " << height << "\n255\n";
    for(int i = 0; i < width*height; ++i){
        file << pixels[i];
    }

    std::cout << "Program done.";

    checkCudaErrors(cudaFree(pixels));
}