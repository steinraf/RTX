#include <iostream>
#include <fstream>
#include <filesystem>

#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "src/Vector.h"
#include "src/Ray.h"
#include "src/Sphere.h"
#include "src/hitable_list.h"
#include "src/Camera.h"
#include "src/material.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ Vector3f color(const Ray& r, Hittable **world, curandState *local_rand_state) {
    Ray cur_ray = r;
    Vector3f cur_attenuation = Vector3f(1.0, 1.0, 1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vector3f attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return Vector3f(0.0, 0.0, 0.0);
            }
        }
        else {
            Vector3f unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            Vector3f c = (1.0f - t) * Vector3f(1.0, 1.0, 1.0) + t * Vector3f(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return Vector3f(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(Vector3f *fb, int max_x, int max_y, int ns, Camera **cam, Hittable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    Vector3f col(0, 0, 0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        Ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

__global__ void create_world(Hittable **d_list, Hittable **d_world, Camera **d_camera, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new Sphere(Vector3f(0, 0, -1), 0.5,
                               new lambertian(Vector3f(0.1, 0.2, 0.5)));
        d_list[1] = new Sphere(Vector3f(0, -100.5, -1), 100,
                               new lambertian(Vector3f(0.8, 0.8, 0.0)));
        d_list[2] = new Sphere(Vector3f(1, 0, -1), 0.5,
                               new metal(Vector3f(0.8, 0.6, 0.2), 0.0));
        d_list[3] = new Sphere(Vector3f(-1, 0, -1), 0.5,
                               new dielectric(1.5));
        d_list[4] = new Sphere(Vector3f(-1, 0, -1), -0.45,
                               new dielectric(1.5));
        *d_world = new hitable_list(d_list,5);
        Vector3f lookfrom(3, 3, 2);
        Vector3f lookat(0, 0, -1);
        float dist_to_focus = (lookfrom-lookat).length();
        float aperture = 2.0;
        *d_camera   = new Camera(lookfrom,
                                 lookat,
                                 Vector3f(0, 1, 0),
                                 20.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(Hittable **d_list, Hittable **d_world, Camera **d_camera) {
    for(int i=0; i < 5; i++) {
        delete ((Sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main() {

    unsigned int width = 192, height = 108;
    unsigned int numSubSamples = 100;
    unsigned int blockSizeX = 8, blockSizeY = 8;

    std::cout << "Rendering a " << width << "x" << height << " image with " << numSubSamples << " samples per pixel ";
    std::cout << "in " << blockSizeX << "x" << blockSizeY << " blocks.\n";

    size_t fb_size = width * height * sizeof(Vector3f);

    // allocate FB
    Vector3f *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, width * height * sizeof(curandState)));

    // make our world of hitables & the Camera
    Hittable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 5*sizeof(Hittable *)));
    Hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hittable *)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));
    create_world<<<1,1>>>(d_list, d_world, d_camera, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(width / blockSizeX + 1, height / blockSizeY + 1);
    dim3 threads(blockSizeX, blockSizeY);
    render_init<<<blocks, threads>>>(width, height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, width, height, numSubSamples, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    const std::string base_path = std::filesystem::path(__FILE__).parent_path();
    const std::string path = base_path + "/data/image.ppm";

    std::ofstream file(path);

    // Output FB as Image
    file << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j * width + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            file << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}
