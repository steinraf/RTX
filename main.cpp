#include <iostream>
#include "src/scene.h"


int main() {
    Scene scene(400, 300);
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{   0,      0, -1}, 0.5));
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{-1.0,      0, -1}, 0.5, 1.0));
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{ 1.0,      0, -1}, 0.5, 1.0));
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{   0, -100.5, -1}, 100));
    scene.render();
    return EXIT_SUCCESS;
}
