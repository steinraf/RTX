#include <iostream>
#include "src/scene.h"

// Benchmark over 100 runs with 192x109 image, 100 subS, 50 maxRay =>
//Mean:	5.61822
//Stddev:	0.0618216
//
//---------
//-----
//--------
//----------------------------
//------------------
//------------------------
//------
//--
//
//

void randomSpheres() {
    Scene scene(3840, 2160, 100, 50);
    for (int i = 0; i < 30; ++i) {
        Color c{getRandom()/2 + 0.5, getRandom()/2 + 0.5, getRandom()/2 + 0.5};
        auto mat = std::make_shared<Metal>(c);
        scene.addShape(
                std::make_shared<Sphere>(
                        Eigen::Vector3d{getRandom(), -0.3 + getRandom() + 1, getRandom() - 2}, 0.2,
                                         mat));
    }
    Color groundColor{getRandom()/2 + 0.5, getRandom()/2 + 0.5, getRandom()/2 + 0.5};
    scene.addShape(
            std::make_shared<Sphere>(Eigen::Vector3d{0, -100.5, -1}, 100, std::make_shared<Lambertian>(groundColor))
            );
    scene.render();
}

void threeSpheres() {
    Scene scene(1920, 1080, 10, 5);
    auto groundMat = std::make_shared<Lambertian>(Color{0.8, 0.8, 0.0});
    auto leftMat = std::make_shared<Metal>(Color{0.8, 0.8, 0.8});
    auto midMat = std::make_shared<Lambertian>(Color{0.7, 0.3, 0.3});
    auto rightMat = std::make_shared<Metal>(Color{0.8, 0.6, 0.2});

    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{-1.0,    0.0, -1.0},   0.5, leftMat  ));
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{ 0.0,    0.0, -1.0},   0.5, midMat   ));
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{ 1.0,    0.0, -1.0},   0.5, rightMat ));
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{ 0.0, -100.5, -1.0}, 100.0, groundMat));

    scene.render();
}


int main() {
    Benchmark b(randomSpheres, 1);
    b.plotHistogram();
    return EXIT_SUCCESS;
}
