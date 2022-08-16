#include <pngwriter.h>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdio.h>



#include <fenv.h>

#include <cuda/std/cassert>

#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "src/Vector.h"
#include "src/Ray.h"
#include "src/sphere.h"
#include "src/hitable_list.h"
#include "src/Camera.h"
#include "src/material.h"
#include "src/scenes.h"
#include "src/util.h"





__device__ Color color(const Ray &r, Hittable **world, curandState *local_rand_state) {
    Ray cur_ray = r;
    const unsigned int maxRayDepth = 50;
    Color cur_attenuation = Color(1.0, 1.0, 1.0);
    for (int i = 0; i < maxRayDepth; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec, local_rand_state)) {
            Ray scattered;
            Color attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return Color(0.0, 0.0, 0.0);
            }
        } else {
//            assert(cur_ray.direction().norm() != 0);
            Vector3f unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction[1] + 1.0f);
            Vector3f c = (1.0f - t) * Vector3f(1.0, 1.0, 1.0) + t * Vector3f(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return Color(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(42, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void
render(Vector3f *fb, int max_x, int max_y, int ns, Camera **cam, Hittable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    Color col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        Ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;

    // TODO find cause for occasional nans
//    for(int i = 0; i < 3; ++i){
//        if (!(col[i] >= 0 and col[i] <= ns)){
//            printf("%f", col[i]);
//            assert(!pixel_index);
//        }
//    }

    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;


}



__global__ void create_world(Hittable **d_list, Hittable **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state) {
//    create_spheres(d_list, d_world, d_camera, nx, ny, rand_state);
    create_cover0(d_list, d_world, d_camera, nx, ny, rand_state);
//    create_room(d_list, d_world, d_camera, nx, ny, rand_state);
}


__global__ void free_world(Hittable **d_list, Hittable **d_world, Camera **d_camera) {
//    delete_spheres(d_list, d_world, d_camera);
    delete_cover0(d_list, d_world, d_camera);
//    delete_room(d_list, d_world, d_camera);


//    auto arr = gpu::array<int, 2>{1, 2};

}

int main() {

//    feenableexcept(FE_INVALID | FE_OVERFLOW);


    unsigned int width = 384, height = 216;
//    unsigned int width = 1920, height = 1080;

    unsigned int numSubSamples = 10;

    unsigned int blockSizeX = 8, blockSizeY = 8;

    std::cout << "Rendering a " << width << "x" << height << " image with " << numSubSamples << " samples per pixel ";
    std::cout << "in " << blockSizeX << "x" << blockSizeY << " blocks.\n";

    size_t fb_size = width * height * sizeof(Vector3f);


    Vector3f *fb;
    curandState *d_rand_state;
    Hittable **d_list, **d_world;
    Camera **d_camera;

    clock_t start, stop;
    start = clock();

    checkCudaErrors(cudaMallocManaged((void **) &fb, fb_size));
    checkCudaErrors(cudaMalloc((void **) &d_rand_state, width * height * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **) &d_list, 487 * sizeof(Hittable *)));
    checkCudaErrors(cudaMalloc((void **) &d_world, sizeof(Hittable *)));
    checkCudaErrors(cudaMalloc((void **) &d_camera, sizeof(Camera *)));



    dim3 blocks(width / blockSizeX + 1, height / blockSizeY + 1);
    dim3 threads(blockSizeX, blockSizeY);

    render_init<<<blocks, threads>>>(width, height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    create_world<<<1, 1>>>(d_list, d_world, d_camera, width, height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(fb, width, height, numSubSamples, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cout << "took " << timer_seconds << " seconds.\n";

    const std::string base_path = std::filesystem::path(__FILE__).parent_path();
    const std::string path = base_path + "/data/image.ppm";
    const std::string png_path = base_path + "/data/image.png";

    pngwriter png(width, height, 1., png_path.c_str());

    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            png.plot(i, j, fb[j*width + i][0], fb[j*width + i][1], fb[j*width + i][2]);
        }
    }

    png.close();



//    std::ofstream file(path);
//
//    // Output FB as Image
//    file << "P3\n" << width << " " << height << "\n255\n";
//    for (int j = height - 1; j >= 0; j--) {
//        for (int i = 0; i < width; i++) {
//            for(int k = 0; k < 3; k++){
//
//                if (int(255.99 * fb[j * width + i][k]) < 0 or int(255.99 * fb[j * width + i][k]) > 255){
//                    std::cout << "Broken thing is at " << fb[j * width + i] << ' ' << i << ' ' << j << '\n';
//                    fb[j * width + i][0] = 1;
//                    fb[j * width + i][1] = 0;
//                    fb[j * width + i][2] = 1;
//                    break;
////                    assert(0);
//                }
//
//            }
//            file << fb[j * width + i];
//        }
//    }

    std::cout << "Drew image to file\n";

    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();

    return EXIT_SUCCESS;
}
