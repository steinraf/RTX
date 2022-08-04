//
// Created by steinraf on 04/08/22.
//

#include "util.h"

__device__ __host__ Vector3f::Vector3f()
        : data{0.0f, 0.0f, 0.0f}{

}

__device__ __host__ Vector3f::Vector3f(FLOAT_TYPE x, FLOAT_TYPE y, FLOAT_TYPE z)
        : data{x, y, z} {
}


__device__ Vector3f::Vector3f(const Vector3f &other)
        : data{other.data[0], other.data[1], other.data[2]} {
}

__device__ Vector3f &Vector3f::operator=(const Vector3f &other) {
    return *this = Vector3f{other};
}

__device__ Vector3f Vector3f::Zero() {
    return Vector3f{0.0f, 0.0f, 0.0f};
}

__device__ Vector3f Vector3f::Random() {
    while(true){
        Vector3f vec{getRandom(), getRandom(), getRandom()};
        if(vec.norm() > 1)
            continue;

        return vec.normalized();
    }
}

__device__ FLOAT_TYPE Vector3f::dot(const Vector3f &other) const {
    FLOAT_TYPE sum = 0.0f;
    for(int i = 0; i < 3; ++i)
        sum += data[i] * other.data[i];
    return sum;
}

__device__ Vector3f Vector3f::cross(const Vector3f &other) const {
    return Vector3f{
            data[1]*other.data[2] - data[2]*other.data[1],
            data[2]*other.data[0] - data[0]*other.data[2],
            data[0]*other.data[1] - data[1]*other.data[0],
    };
}

__device__ inline Vector3f Vector3f::operator-(const Vector3f &other) const {
    FLOAT_TYPE arr[3];
    for(int i = 0; i < 3; ++i)
        arr[i] = data[i] - other.data[i];
    return Vector3f{arr[0], arr[1], arr[2]};
}

__device__ Vector3f Vector3f::operator+(const Vector3f& other) const{
    FLOAT_TYPE arr[3];
    for(int i = 0; i < 3; ++i)
        arr[i] = data[i] + other.data[i];
    return Vector3f{arr[0], arr[1], arr[2]};
}

__device__ Vector3f& Vector3f::operator+=(const Vector3f& other) {
    for(int i = 0; i < 3; ++i)
        data[i] += other.data[i];
    return *this;
}

__device__ inline Vector3f Vector3f::operator*(FLOAT_TYPE scalar) const{
    FLOAT_TYPE arr[3];
    for(int i = 0; i < 3; ++i)
        arr[i] = data[i] * scalar;
    return Vector3f{arr[0], arr[1], arr[2]};
}

__device__ Vector3f operator*(FLOAT_TYPE scalar, const Vector3f &vec) {
    return vec * scalar;
}

__device__ inline Vector3f Vector3f::operator/(FLOAT_TYPE scalar) const {
    FLOAT_TYPE arr[3];
    for(int i = 0; i < 3; ++i)
        arr[i] = data[i]/scalar;
    return Vector3f{arr[0], arr[1], arr[2]};
}

__device__ __host__ FLOAT_TYPE Vector3f::operator[](size_t idx) const{
    return data[idx];
}


__device__ inline FLOAT_TYPE Vector3f::squaredNorm() const {
    return this->dot(*this);
}

__device__ inline FLOAT_TYPE Vector3f::norm() const {
    return std::sqrt(this->squaredNorm());
}

__device__ inline Vector3f Vector3f::normalized() const {
    return *this/this->norm();
}

__host__ std::ostream& operator<<(std::ostream& out, Color vec){
    out << static_cast<int>(255.99 * vec[0]) << ' '
        << static_cast<int>(255.99 * vec[1]) << ' '
        << static_cast<int>(255.99 * vec[2]) << '\n';
    return out;
}


__device__ __host__ FLOAT_TYPE getRandom(){

#ifndef __CUDA__ARCH__
        return 0.01f;
#endif
        return 0.1f;
}



__device__ Ray::Ray(Vector3f pos, Vector3f dir)
        : pos(pos), dir(dir.normalized()) {
}

__device__ Vector3f Ray::at(FLOAT_TYPE dist) const {
    return pos + dist * dir;
}

__device__ inline Vector3f getRandomInSphere() {
    while (true) {
        Vector3f v{getRandom(), getRandom(), 0};
        if (v.norm() <= 1) return v;
    }
}
