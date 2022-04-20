//
// Created by steinraf on 20/04/2022.
//

#include "materials.h"

Lambertian::Lambertian(double reflectivity)
        : reflectivity(reflectivity){

}

std::pair<Ray, double> Lambertian::scatter(const Ray &ray, const Hit &hit) const {
    Eigen::Vector3d target = ray.pos + hit.normal + ((1 - reflectivity) * Eigen::Vector3d::Random()).normalized();
    Ray r{ray.pos, (target - ray.pos).normalized()};
    return {r, reflectivity};
}

