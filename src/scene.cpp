//
// Created by steinraf on 16/04/2022.
//

#include "scene.h"

Scene::Scene(unsigned int width, unsigned int height, unsigned int subSamples, unsigned int maxRayDepth)
        : width(width), height(height), camera(width, height), subSamples(subSamples), maxRayDepth(maxRayDepth) {

}

void Scene::render() {


    std::vector<Color> colors(width * height);

    std::atomic<int> counter;
#pragma omp parallel for shared(counter, colors, std::cerr), default(none)
    for (unsigned int j = 0; j < height; ++j) {
        std::cerr << "\rScanlines remaining: " << height - ++counter << ' ' << std::flush;
        for (unsigned int i = 0; i < width; ++i) {
            Color c;
            c.subSamples = subSamples;
            for (int s = 0; s < subSamples; ++s) {
                auto ray = camera.getRay(i + getRandom(), j + getRandom());
                c += castRay(ray, maxRayDepth);
            }
            colors[j * width + i] = c;
        }
    }
//    std::cout << '\n';

    const std::vector<char> characters = {
            ' ',
            '.',
            ',',
            '-',
            '~',
            ':',
            ';',
            '=',
            '!',
            '*',
            '#',
            '&',
            '@',
    };

//        for(int j = 0; j < height; ++j){
//            for(int i = 0; i < width; ++i){
//                double idx = colors[j*width+i].asGray()/colors[j*width+i].subSamples*characters.size();
//    //          std::cout << idx << ' ';
//                std::cout << characters[static_cast<unsigned>(idx)];
//            }
//            std::cout << '\n';
//        }

    std::ofstream file(path);
    file << "P3\n"
         << width << ' ' << height << "\n"
                                      "255\n";

    for (auto c: colors) {
        file << c;
    }
}

void Scene::addShape(std::shared_ptr<Shape> shape) {
    shapes.push_back(shape);

}

std::pair<std::shared_ptr<Shape>, double> Scene::getClosestIntersect(const Ray &r) const {
    std::pair<std::shared_ptr<Shape>, double> best = {nullptr, std::numeric_limits<double>::max()};
    for (auto shape: shapes) {
        double d = shape->findIntersect(r);
        if (d >= 0.001 && d < best.second)
            best = {shape, d};
    }
    return best;
}

Color Scene::getBackground(const Ray &ray) const {

    double grad = (ray.dir[1] + 1) / 2; // Map from -1 1 to 0 1
    return (1 - grad) * Color{1.0, 1.0, 1.0} + grad * Color{0.5, 0.7, 1.0};
}

Color Scene::castRay(const Ray &ray, int rayDepth) const {
    if (rayDepth <= 0)
        return Color{0, 0, 0};

    const auto &[obj, dist] = getClosestIntersect(ray);

    if (obj) {
        auto hit = obj->calculateHit(ray, dist);
        auto [newRay, attenuation] = obj->scatter(ray, hit);
        return attenuation * castRay(newRay, rayDepth - 1);
    }

    return getBackground(ray);
}
