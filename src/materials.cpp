//
// Created by steinraf on 20/04/2022.
//

#include "materials.h"

Lambertian::Lambertian(const Color &reflectivity)
        : reflectivity(reflectivity){

}

std::pair<Ray, Color> Lambertian::scatter(const Ray &ray, const Hit &hit) const {
    Eigen::Vector3d scatterDir = hit.normal + Eigen::Vector3d::Random().normalized();
    if(scatterDir.norm() < 1e-4)
        scatterDir = hit.normal;
    Ray r{hit.intersectPos(), scatterDir.normalized()};
    return {r, reflectivity};
}

Metal::Metal(const Color &reflectivity)
    :reflectivity(reflectivity){

}

std::pair<Ray, Color> Metal::scatter(const Ray &ray, const Hit &hit) const {
    Ray r{
        hit.intersectPos(),
        (ray.dir - 2*ray.dir.dot(hit.normal)*hit.normal).normalized()
    };
    //assert(r.dir.dot(hit.normal) > 0);
    return {r, reflectivity};
}
