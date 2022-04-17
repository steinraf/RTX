//
// Created by steinraf on 16/04/2022.
//

#pragma once

#include "util.h"
#include "shapes.h"
#include <fstream>
#include <iostream>


class Scene {
public:
    Scene(unsigned int width, unsigned int height, unsigned int subSamples=100, unsigned int maxRayDepth=50);

    void render();

    void addShape(std::shared_ptr<Shape> shape);

private:


    unsigned int width;
    unsigned int height;

    Camera camera;

    const int subSamples;
    const int maxRayDepth;

    std::pair<std::shared_ptr<Shape>, double> getClosestIntersect(const Ray& r) const;

    std::vector<std::shared_ptr<Shape>> shapes;

    Color getBackground(const Ray& ray) const;

    Color castRay(const Ray& ray, int maxRayDepth) const;

    const std::string path = "/home/steinraf/Coding/Raytracing/data/image.ppm";

};


