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

void runRenderer() {
    Scene scene(192, 108, 100, 50);
    for (int i = 0; i < 30; ++i) {
        auto mat = std::make_shared<Lambertian>(getRandom()/2 + 0.5);
        scene.addShape(
                std::make_shared<Sphere>(
                        Eigen::Vector3d{getRandom(), -0.3 + getRandom() + 1, getRandom() - 2}, 0.2,
                                         mat));
    }
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{0, -100.5, -1}, 100, std::make_shared<Lambertian>(getRandom()/2 + 0.5)));
    scene.render();
}


int main() {
    Benchmark b(runRenderer, 1);
    b.plotHistogram();
    return EXIT_SUCCESS;
}
