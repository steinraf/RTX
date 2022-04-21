#include <iostream>
#include "src/scene.h"

//HD 1920, 2080
//4K 3840, 2160

void sphereLineup(){
    Scene scene(3840, 2160, 100, 50);

    auto groundMat = std::make_shared<Lambertian>(Color{0.8, 0.8, 0.0});
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{ 0.0, -1000.5, -1.0}, 1000.0, groundMat));


    auto glassMat = std::make_shared<Dielectric>(Color{1.0, 1.0, 1.0}, 1.5);

    for(int i = 0; i < 10; ++i){
        Color c{i/10.0, 1 - i/10.0, 0.4};
        auto mat = std::make_shared<Metal>(c, 0.3);

        scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{ (i-5)*1.0,   0, -5},   0.5, mat   ));
    }

    for(int i = 0; i < 10; ++i){
        Color c{i/10.0, 1 - i/10.0, 0.4};
        auto mat = std::make_shared<Metal>(c, 0.3);

        scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{ (i-5)*1.0,   4.0, -5},   0.5, mat   ));
    }

    auto midMat = std::make_shared<Dielectric>(Color{1.0, 1.0, 0.7}, 1.5);
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{ 0.0,    0.0, -1.0},   0.5, midMat   ));


    scene.render();
}

void threeSpheres() {
    Scene scene(400, 300, 100, 50);
    auto groundMat = std::make_shared<Lambertian>(Color{0.8, 0.8, 0.0});
    auto leftMat = std::make_shared<Lambertian>(Color{0.7, 0.4, 0.8});
    auto midMat = std::make_shared<Dielectric>(Color{1.0, 1.0, 0.7}, 1.5);
    auto rightMat = std::make_shared<Metal>(Color{0.7, 1.0, 0.7}, 1.5);

    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{ 0.0, -1000.5, -1.0}, 1000.0, groundMat));


    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{-1.0,    0.0, -1.0},   0.5, leftMat  ));
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{ 0.0,    0.0, -1.0},   0.5, midMat   ));
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{ 1.0,    0.0, -1.0},   0.5, rightMat ));

    scene.render();
}


int main() {
    Benchmark b(sphereLineup, 1);
    b.plotHistogram();
    return EXIT_SUCCESS;
}
