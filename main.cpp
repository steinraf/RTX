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

void runRenderer(){
    Scene scene(192, 108, 100, 50);
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{   0,      0, -1}, 0.5));
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{-1.0,      0, -1}, 0.5, 1.0));
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{ 1.0,      0, -1}, 0.5, 1.0));
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{   0, -100.5, -1}, 100));
    scene.render();


}


int main() {
    Benchmark b(runRenderer, 100);
    b.plotHistogram();
    return EXIT_SUCCESS;
}
