#include <iostream>
#include "src/scene.h"

//HD 1920, 2080
//4K 3840, 2160

void sphereLineup(){
    Scene scene(400, 300, 100, 50);

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


void randomScene(){
    Scene scene(3840, 2160, 100, 50);

    auto groundMat = std::make_shared<Lambertian>(Color{0.5, 0.5, 0.5});
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{0, -1000,0},    1000, groundMat));

    for (int x = -11; x < 11; ++x) {
        for (int z = -11; z < 11; z++) {
            auto rand = getRandom()/2 + 0.5;
            Eigen::Vector3d pos{x + 0.9*(getRandom()/2 + 0.5), 0.2, z + 0.9*(getRandom()/2 + 0.5)};

            if((pos - Eigen::Vector3d{4, 0.2, 0}).norm() > 0.9){
                std::shared_ptr<Material> mat;
                if(rand < 0.8) {
                    Color c{(getRandom()/2 + 0.5)*(getRandom()/2 + 0.5), (getRandom()/2 + 0.5)*(getRandom()/2 + 0.5), (getRandom()/2 + 0.5)*(getRandom()/2 + 0.5)};
                    mat = std::make_shared<Lambertian>(c);
                    scene.addShape(std::make_shared<Sphere>(pos, 0.2, mat));
                } else if(rand < 0.95) {
                    Color c{(getRandom()/4 + 0.75), (getRandom()/4 + 0.75), (getRandom()/4 + 0.75)};
                    double fuzziness = (getRandom()/4 + 0.25);
                    mat = std::make_shared<Metal>(c, fuzziness);
                    scene.addShape(std::make_shared<Sphere>(pos, 0.2, mat));
                } else {
                    mat = std::make_shared<Dielectric>(Color{1.0, 1.0, 1.0}, 1.5);
                    scene.addShape(std::make_shared<Sphere>(pos, 0.2, mat));
                }
            }
        }
    }

    auto mat1 = std::make_shared<Dielectric>(Color{1.0, 1.0, 1.0}, 1.5);
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{0, 1, 0}, 1.0, mat1));

    auto mat2 = std::make_shared<Lambertian>(Color{0.4, 0.2, 0.1});
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{-4, 1, 0}, 1.0, mat2));

    auto mat3 = std::make_shared<Metal>(Color{0.7, 0.6, 0.5}, 0.0);
    scene.addShape(std::make_shared<Sphere>(Eigen::Vector3d{4, 1, 0}, 1.0, mat3));

    scene.render();

}


int main() {
    Benchmark b(randomScene, 1);
    b.plotHistogram();
    return EXIT_SUCCESS;
}
