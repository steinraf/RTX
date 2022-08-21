//
// Created by steinraf on 21/08/22.
//

#pragma once

#include "vector.h"
#include "ray.h"

struct HitRecord{
    Vector3f position;
    Vector3f normal;
    double t;
    bool frontFace;

    __device__ inline void setFaceNormal(const Ray& r, const Vector3f& outwardNormal){
        frontFace = r.getDirection().dot(outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};


class Hittable {
public:
    __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const = 0;
};


