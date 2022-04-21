//
// Created by steinraf on 16/04/2022.
//

#include "shapes.h"


Shape::Shape(std::shared_ptr<Material> mat)
        : material(mat){

}

Sphere::Sphere(Eigen::Vector3d pos, double r, std::shared_ptr<Material> mat)
        : Shape(mat), center(pos), radius(r) {

}

double Sphere::findIntersect(const Ray &ray) const {
    const double a = ray.dir.dot(ray.pos - center);
    const double discriminant = std::pow(a, 2) - ((ray.pos - center).squaredNorm() - std::pow(radius, 2));
    if (discriminant < 0) // No intersection
        return -1.0;
    else if (discriminant == 0) // Line perfectly touches line (very very unlikely b.o. num. stab.)
        return -a;
    else {
        const double sqrDisc = std::sqrt(discriminant);
        const double sol1 = -a - sqrDisc;
        const double sol2 = -a + sqrDisc;
        const double min = std::min(sol1, sol2);
        if (min < 0.001)
            return std::max(sol1, sol2);
        else
            return min;
    }
}

Eigen::Vector3d Sphere::getNormal(const Eigen::Vector3d &pos) const {
    return (pos - center).normalized();
}

Hit Sphere::calculateHit(const Ray &ray, double intersect) {
    return {
            ray,
            intersect,
            getNormal(ray.at(intersect))
    };
}


