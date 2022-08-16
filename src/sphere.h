#pragma once

#include "Hittable.h"

class Sphere : public Hittable {
public:
    __device__ Sphere() {}

    __device__ Sphere(Vector3f cen, float r, material *m) : center(cen), radius(r), mat_ptr(m) {};

    __device__ virtual bool hit(const Ray &r, float tmin, float tmax, hit_record &rec, curandState * rand) const;

    Vector3f center;
    float radius;
    material *mat_ptr;
};


