//
// Created by steinraf on 21/08/22.
//

#pragma once

#include "ray.h"
#include "hittable.h"


class Sphere : public Hittable {
public:
    Sphere() = delete;
    __device__ Sphere(const Vector3f& center, double radius) : center(center), radius(radius){}


    __device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;

private:
    Vector3f center;
    float radius;
};


