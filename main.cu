#include <iostream>
#include <filesystem>
#include <fstream>


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit( result);
    }
}

__global__ void render(float *fb, size_t max_x, size_t max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x*3 + i*3;
    fb[pixel_index + 0] = float(i) / max_x;
    fb[pixel_index + 1] = float(j) / max_y;
    fb[pixel_index + 2] = float(i) * float(j) / max_x / max_y;
}

int main() {

    size_t width = 3840, height = 2160;
    size_t block_x = 8, block_y = 8;

    float *pixels;
    checkCudaErrors(cudaMallocManaged((void **)&pixels, 3*width*height*sizeof(float)));


    dim3 blocks(width/block_x+1,height/block_y+1);
    dim3 threads(block_x,block_y);
    render<<<blocks, threads>>>(pixels, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    const std::string current_path = std::filesystem::path(__FILE__).parent_path();
    const std::string path = current_path + "/data/image.ppm";

    std::ofstream file{path};

    // Output FB as Image
    file << "P3\n" << width << " " << height << "\n255\n";
    for(int i = 0; i < width*height; ++i){
        file << int(255.99 * pixels[3*i    ]) << " "
             << int(255.99 * pixels[3*i + 1]) << " "
             << int(255.99 * pixels[3*i + 2]) << "\n";
    }



    checkCudaErrors(cudaFree(pixels));
}