//
// Created by steinraf on 20/04/2022.
//

#include "materials.h"


Lambertian::Lambertian(const Color &reflectivity)
        : reflectivity(reflectivity){

}

std::pair<Ray, Color> Lambertian::scatter(const Ray &ray, const Hit &hit) const {
    Vector3f scatterDir = hit.normal + Vector3f::Random().normalized();
    if(scatterDir.norm() < 1e-4)
        scatterDir = hit.normal;
    Ray r{hit.intersectPos(), scatterDir.normalized()};
    return {r, reflectivity};
}

Metal::Metal(const Color &reflectivity, double fuzziness)
    :reflectivity(reflectivity), fuzziness(fuzziness){

}

std::pair<Ray, Color> Metal::scatter(const Ray &ray, const Hit &hit) const {
    Ray r{
        hit.intersectPos(),
        ray.dir - 2*ray.dir.dot(hit.normal)*hit.normal + Vector3f::Random().normalized() * fuzziness
    };
    //assert(r.dir.dot(hit.normal) > 0);
    return {r, reflectivity};
}

Dielectric::Dielectric(const Color &reflectivity, double refractiveIndex)
    :reflectivity(reflectivity), refractiveIndex(refractiveIndex){

}

std::pair<Ray, Color> Dielectric::scatter(const Ray &ray, const Hit &hit) const {

    double cosTheta = -ray.dir.dot(hit.normal);
    double sinTheta = std::sqrt(1 - std::pow(cosTheta, 2));


    //Checks if ray comes from inside the object
    double refract_ratio = cosTheta < 0 ? refractiveIndex : 1.0/refractiveIndex;

    if(refract_ratio * sinTheta >= 1.0){
        Ray r{
                hit.intersectPos(),
                ray.dir - 2*ray.dir.dot(hit.normal)*hit.normal
        };
        return {r, reflectivity};
    }else{
        int sign = cosTheta > 0 ? 1 : -1;
        Vector3f orthDir = refract_ratio * (ray.dir + cosTheta*hit.normal);
        Vector3f paraDir = -sign*std::sqrt(std::abs(1.0 - orthDir.squaredNorm())) * hit.normal;

        Ray r{
                hit.intersectPos(),
                orthDir + paraDir
        };
        return {r, reflectivity};
    }
}
