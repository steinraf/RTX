#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cuda/std/cassert>
#include <curand_kernel.h>

class Vector3f {


public:
    __host__ __device__ Vector3f() : data{0.0f, 0.0f, 0.0f} {}

    __host__ __device__ Vector3f(float x, float y, float z) : data{x, y, z} {}
    __host__ __device__ Vector3f(float v) : data{v, v, v} {}

    __host__ __device__ inline float operator[](int i) const { return data[i]; }

    __host__ __device__ inline float &operator[](int i) { return data[i]; };

    __host__ __device__ inline Vector3f operator-() const;

    __host__ __device__ inline Vector3f operator+(const Vector3f &v2) const;

    __host__ __device__ inline Vector3f operator-(const Vector3f &v2) const;

    __host__ __device__ inline Vector3f operator*(const Vector3f &v2) const;

    __host__ __device__ inline Vector3f operator/(const Vector3f &v2) const;

    __host__ __device__ inline Vector3f operator*(float t) const;

    __host__ __device__ inline Vector3f operator/(float t) const;

    __host__ __device__ inline Vector3f &operator+=(const Vector3f &v2);

    __host__ __device__ inline Vector3f &operator-=(const Vector3f &v2);

    __host__ __device__ inline Vector3f &operator*=(const Vector3f &v2);

    __host__ __device__ inline Vector3f &operator/=(const Vector3f &v2);

    __host__ __device__ inline Vector3f &operator*=(float t);

    __host__ __device__ inline Vector3f &operator/=(float t);

    __host__ __device__ inline float norm() const;

    __host__ __device__ inline float squaredNorm() const;

    __host__ __device__ inline Vector3f normalized() const;

    __host__ __device__ inline float dot(const Vector3f &v2) const;

    __host__ __device__ inline Vector3f cross(const Vector3f &v2) const;

    __host__ __device__ inline int asColor(size_t i) const;

    __device__ static inline Vector3f Random(curandState *local_rand_state);

    __device__ static inline Vector3f RandomInUnitDisk(curandState *local_rand_state);
    __device__ static inline Vector3f RandomInUnitSphere(curandState *local_rand_state);

    friend inline std::ostream &operator<<(std::ostream &os, const Vector3f &t);



private:
    float data[3];
};

using Color = Vector3f;


inline std::ostream &operator<<(std::ostream &os, const Vector3f &t) {
    os << t.asColor(0) << ' '
       << t.asColor(1) << ' '
       << t.asColor(2) << '\n';
    return os;
}


__host__ __device__ inline int Vector3f::asColor(size_t i) const {
    return int(255.99 * data[i]);
}

__host__ __device__ inline float Vector3f::norm() const {
    return std::sqrt(squaredNorm());
}

__host__ __device__ inline float Vector3f::squaredNorm() const {
    return data[0] * data[0]
           + data[1] * data[1]
           + data[2] * data[2];
}


__host__ __device__ inline Vector3f Vector3f::normalized() const {
    float n = norm();
    assert(n != 0);
    float k = 1.0 / n;
    return Vector3f(
            data[0] * k,
            data[1] * k,
            data[2] * k
    );
}

__host__ __device__ inline Vector3f Vector3f::operator-() const {
    return Vector3f(-data[0], -data[1], -data[2]);
}


__host__ __device__ inline Vector3f Vector3f::operator+(const Vector3f &v2) const {
    return Vector3f(data[0] + v2.data[0],
                    data[1] + v2.data[1],
                    data[2] + v2.data[2]);
}

__host__ __device__ inline Vector3f Vector3f::operator-(const Vector3f &v2) const {
    return Vector3f(data[0] - v2.data[0],
                    data[1] - v2.data[1],
                    data[2] - v2.data[2]);
}

__host__ __device__ inline Vector3f Vector3f::operator*(const Vector3f &v2) const {
    return Vector3f(data[0] * v2.data[0],
                    data[1] * v2.data[1],
                    data[2] * v2.data[2]);
}

__host__ __device__ inline Vector3f Vector3f::operator/(const Vector3f &v2) const {
    assert(v2.data[0] != 0);
    assert(v2.data[1] != 0);
    assert(v2.data[2] != 0);
    return Vector3f(data[0] / v2.data[0],
                    data[1] / v2.data[1],
                    data[2] / v2.data[2]);
}

__host__ __device__ inline Vector3f operator*(float t, const Vector3f &v) {
    return v * t;
}

__host__ __device__ inline Vector3f Vector3f::operator*(float t) const {
    return Vector3f(t * data[0],
                    t * data[1],
                    t * data[2]);
}




__host__ __device__ inline Vector3f Vector3f::operator/(float t) const {
//    assert(t != 0);
    return Vector3f(data[0] / t,
                    data[1] / t,
                    data[2] / t);
}


__host__ __device__ inline float Vector3f::dot(const Vector3f &v2) const {
    return data[0] * v2.data[0] +
           data[1] * v2.data[1] +
           data[2] * v2.data[2];
}

__host__ __device__ inline Vector3f Vector3f::cross(const Vector3f &v2) const {
    return Vector3f(data[1] * v2.data[2] - data[2] * v2.data[1],
                    data[2] * v2.data[0] - data[0] * v2.data[2],
                    data[0] * v2.data[1] - data[1] * v2.data[0]);
}


__host__ __device__ inline Vector3f &Vector3f::operator+=(const Vector3f &v) {
    data[0] += v.data[0];
    data[1] += v.data[1];
    data[2] += v.data[2];
    return *this;
}

__host__ __device__ inline Vector3f &Vector3f::operator*=(const Vector3f &v) {
    data[0] *= v.data[0];
    data[1] *= v.data[1];
    data[2] *= v.data[2];
    return *this;
}

__host__ __device__ inline Vector3f &Vector3f::operator/=(const Vector3f &v) {
    assert(v.data[0] != 0);
    assert(v.data[1] != 0);
    assert(v.data[2] != 0);

    data[0] /= v.data[0];
    data[1] /= v.data[1];
    data[2] /= v.data[2];
    return *this;
}

__host__ __device__ inline Vector3f &Vector3f::operator-=(const Vector3f &v) {
    data[0] -= v.data[0];
    data[1] -= v.data[1];
    data[2] -= v.data[2];
    return *this;
}

__host__ __device__ inline Vector3f &Vector3f::operator*=(float t) {
    data[0] *= t;
    data[1] *= t;
    data[2] *= t;
    return *this;
}

__host__ __device__ inline Vector3f &Vector3f::operator/=(float t) {
    assert(t != 0);
    float k = 1.0 / t;

    data[0] *= k;
    data[1] *= k;
    data[2] *= k;
    return *this;
}

__host__ __device__ inline Vector3f unit_vector(Vector3f v) {
    auto n = v.norm();
//    assert(n != 0);
    return v / n;
}

__device__ inline Vector3f Vector3f::Random(curandState *local_rand_state) {
    return Vector3f(curand_uniform(local_rand_state), curand_uniform(local_rand_state),
                    curand_uniform(local_rand_state));
}

__device__ inline Vector3f Vector3f::RandomInUnitDisk(curandState *local_rand_state){
    Vector3f p;
    do {
        p = 2.0f * Vector3f(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - Vector3f(1, 1, 0);
    } while (p.squaredNorm() >= 1.0f);
    return p;
}

__device__ inline Vector3f Vector3f::RandomInUnitSphere(curandState *local_rand_state){
    Vector3f p;
    do {
        p = 2.0f * Vector3f::Random(local_rand_state) - Vector3f(1, 1, 1);
    } while (p.squaredNorm() >= 1.0f);
    return p;
}

