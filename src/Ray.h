#pragma once

#include "Vector.h"

class Ray {
public:
    __device__ Ray():A{1.0, 0.0, 0.0}, B{1.0, 0.0, 0.0} {}

    __device__ Ray(const Vector3f &a, const Vector3f &b) {
        A = a;
        B = b;
    }

    __device__ Vector3f origin() const { return A; }

    __device__ Vector3f direction() const { return B; }

    __device__ Vector3f point_at_parameter(float t) const { return A + t * B; }

    Vector3f A;
    Vector3f B;
};
