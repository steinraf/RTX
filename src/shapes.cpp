//
// Created by steinraf on 16/04/2022.
//

#include "shapes.h"

Sphere::Sphere(Eigen::Vector3d pos, double r, double reflectivity)
    :reflectivity(reflectivity), pos(pos), radius(r){

}

double Sphere::findIntersect(const Ray &ray) const{
    const double a = ray.dir.dot(ray.pos - pos);
    const double discriminant = std::pow(a, 2) - ((ray.pos - pos).squaredNorm() - std::pow(radius, 2));
    if(discriminant < 0) // No intersection
        return -1.0;
    else if (discriminant == 0) // Line perfectly touches line (very very unlikely b.o. num. stab.)
        return -a;
    else{
        const double sqrDisc = std::sqrt(discriminant);
        const double sol1 = -a - sqrDisc;
        const double sol2 = -a + sqrDisc;
        const double min = std::min(sol1, sol2);
        if(min < 0)
            return std::max(sol1, sol2);
        else
            return min;
    }
}

Ray Sphere::reflect(const Ray &ray) const {
    Eigen::Vector3d normal = (ray.pos - pos).normalized();
    Eigen::Vector3d target = ray.pos + normal + ((1-reflectivity)*Eigen::Vector3d::Random()).normalized();
    Ray r{ray.pos, (target - ray.pos).normalized()};
    return r;
}
