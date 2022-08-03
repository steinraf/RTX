#include <iostream>
#include "src/scene.h"

//HD 1920, 2080
//4K 3840, 2160

void sphereLineup(){
    Scene scene(400, 300, 100, 50);

    auto groundMat = new Lambertian(Color{0.8, 0.8, 0.0});
    scene.addShape(new Sphere{Vector3f{ 0.0, -1000.5, -1.0}, 1000.0, groundMat});


    auto glassMat = std::make_shared<Dielectric>(Color{1.0, 1.0, 1.0}, 1.5);

    for(int i = 0; i < 10; ++i){
        Color c{i/10.0f, 1 - i/10.0f, 0.4};
        auto mat = new Metal(c, 0.3f);

        scene.addShape(new Sphere(Vector3f{ (i-5)*1.0f,   0, -5},   0.5, mat   ));
    }

    for(int i = 0; i < 10; ++i){
        Color c{i/10.0f, 1 - i/10.0f, 0.4};
        auto mat = new Metal(c, 0.3);

        scene.addShape(new Sphere(Vector3f{ (i-5)*1.0f,   4.0, -5},   0.5, mat   ));
    }

    auto midMat = new Dielectric(Color{1.0, 1.0, 0.7}, 1.5);
    scene.addShape(new Sphere(Vector3f{ 0.0,    0.0, -1.0},   0.5, midMat   ));


    scene.render();
}

void threeSpheres() {
    Scene scene(400, 300, 100, 50);
    auto groundMat = new Lambertian(Color{0.8, 0.8, 0.0});
    auto leftMat = new Lambertian(Color{0.7, 0.4, 0.8});
    auto midMat = new Dielectric(Color{1.0, 1.0, 0.7}, 1.5);
    auto rightMat = new Metal(Color{0.7, 1.0, 0.7}, 1.5);

    scene.addShape(new Sphere(Vector3f{ 0.0, -1000.5, -1.0}, 1000.0, groundMat));


    scene.addShape(new Sphere(Vector3f{-1.0,    0.0, -1.0},   0.5, leftMat  ));
    scene.addShape(new Sphere(Vector3f{ 0.0,    0.0, -1.0},   0.5, midMat   ));
    scene.addShape(new Sphere(Vector3f{ 1.0,    0.0, -1.0},   0.5, rightMat ));

    scene.render();
}

__global__ void cuda_render(){

}


void randomScene(){
    Scene scene(384, 216, 10, 5);

    auto groundMat = new Lambertian(Color{0.5, 0.5, 0.5});
    scene.addShape(new Sphere(Vector3f{0, -1000,0},    1000, groundMat));

    for (int x = -11; x < 11; ++x) {
        for (int z = -11; z < 11; z++) {
            auto rand = getRandom()/2 + 0.5f;
            Vector3f pos{x + 0.9f*(getRandom()/2 + 0.5f), 0.2, z + 0.9f*(getRandom()/2 + 0.5f)};

            if((pos - Vector3f{4, 0.2, 0}).norm() > 0.9){
                Material *mat;
                if(rand < 0.8) {
                    Color c{(getRandom()/2 + 0.5f)*(getRandom()/2 + 0.5f), (getRandom()/2 + 0.5f)*(getRandom()/2 + 0.5f), (getRandom()/2 + 0.5f)*(getRandom()/2 + 0.5f)};
                    mat = new Lambertian(c);
                    scene.addShape(new Sphere(pos, 0.2, mat));
                } else if(rand < 0.95) {
                    Color c{(getRandom()/4 + 0.75f), (getRandom()/4 + 0.75f), (getRandom()/4 + 0.75f)};
                    double fuzziness = (getRandom()/4 + 0.25);
                    mat = new Metal(c, fuzziness);
                    scene.addShape(new Sphere(pos, 0.2, mat));
                } else {
                    mat = new Dielectric(Color{1.0, 1.0, 1.0}, 1.5);
                    scene.addShape(new Sphere(pos, 0.2, mat));
                }
            }
        }
    }

    auto mat1 = new Dielectric(Color{1.0, 1.0, 1.0}, 1.5);
    scene.addShape(new Sphere(Vector3f{0, 1, 0}, 1.0, mat1));

    auto mat2 = new Lambertian(Color{0.4, 0.2, 0.1});
    scene.addShape(new Sphere(Vector3f{-4, 1, 0}, 1.0, mat2));

    auto mat3 = new Metal(Color{0.7, 0.6, 0.5}, 0.0);
    scene.addShape(new Sphere(Vector3f{4, 1, 0}, 1.0, mat3));

    scene.render();

}

__global__ void cudaSimpleScene(Shape**shapeBuilder, Shape**shapes, Camera** camera){

    if (!(threadIdx.x == 0 && blockIdx.x == 0))
        return;

    shapeBuilder[0] = new Sphere(Vector3f(0, 0, -1), 0.5, new Lambertian(Color(0.5, 0.2, 0.1)));
}

void cuda_main(){

    size_t width = 3840, height = 2160, num_obj = 1;

    Color *color_buf;

    cudaMallocManaged((void **)&color_buf, width*height * sizeof(Color));

    Shape **shapeBuilder;

    Shape **shapes;

    cudaMallocManaged((void **)&shapeBuilder, num_obj * sizeof(Shape *));
    cudaMallocManaged((void **)&shapes, sizeof(Shape *));

    Camera **cam;

    cudaMallocManaged((void **)&cam, sizeof(Camera *));


    cudaSimpleScene<<<1, 1>>>(shapeBuilder, shapes, cam);

    return;

}

int main() {
//    cuda_main();
//    return EXIT_SUCCESS;
    Benchmark b(randomScene, 1);
    b.plotHistogram();
    return EXIT_SUCCESS;
}
