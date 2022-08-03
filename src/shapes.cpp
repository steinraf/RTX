//
// Created by steinraf on 16/04/2022.
//

#include "shapes.h"


Shape::Shape(std::shared_ptr<Material> mat)
        : material(mat){

}

Sphere::Sphere(Vector3f pos, float r, std::shared_ptr<Material> mat)
        : Shape(mat), center(pos), radius(r) {

}

float Sphere::findIntersect(const Ray &ray) const {
    const float a = ray.dir.dot(ray.pos - center);
    const float discriminant = std::pow(a, 2) - ((ray.pos - center).squaredNorm() - std::pow(radius, 2));
    if (discriminant < 0) // No intersection
        return -1.0;
    else if (discriminant == 0) // Line perfectly touches line (very very unlikely b.o. num. stab.)
        return -a;
    else {
        const float sqrDisc = std::sqrt(discriminant);
        const float sol1 = -a - sqrDisc;
        const float sol2 = -a + sqrDisc;
        const float min = std::min(sol1, sol2);
        if (min < 0.001)
            return std::max(sol1, sol2);
        else
            return min;
    }
}

Vector3f Sphere::getNormal(const Vector3f &pos) const {
    return (pos - center).normalized();
}

Hit Sphere::calculateHit(const Ray &ray, float intersect) {
    return {
            ray,
            intersect,
            getNormal(ray.at(intersect))
    };
}


