//
// Created by steinraf on 04/08/22.
//

#pragma once

#include <ostream>

#define FLOAT_TYPE float
//#define float raise std::runtime_error("Please use FLOAT_TYPE instead of float");float
//#define double raise std::runtime_error("Please use FLOAT_TYPE instead of double");double


class Vector3f{
    using Color = Vector3f;
public:
    __device__ __host__ Vector3f();
    __device__ __host__ Vector3f(FLOAT_TYPE x, FLOAT_TYPE y, FLOAT_TYPE z);
    ~Vector3f() = default;
    __device__ Vector3f(const Vector3f& other);
    Vector3f(Vector3f&& other) noexcept = default;
    __device__ Vector3f& operator=(const Vector3f& other);
    Vector3f& operator=(Vector3f&& other) = default;

    [[nodiscard]] __device__ inline Vector3f operator-(const Vector3f& other) const;

    [[nodiscard]] __device__ Vector3f operator+(const Vector3f& other) const;
    __device__ Vector3f& operator+=(const Vector3f& other);

    friend __device__ Vector3f operator*(FLOAT_TYPE s, const Vector3f& vec);
    [[nodiscard]] __device__ inline Vector3f operator*(FLOAT_TYPE scalar) const;

    [[nodiscard]] __device__ inline Vector3f operator/(FLOAT_TYPE scalar) const;

    [[nodiscard]] __device__ __host__ FLOAT_TYPE operator[](size_t idx) const noexcept(false);

    [[nodiscard]] __device__ inline FLOAT_TYPE squaredNorm() const;

    [[nodiscard]] __device__ inline FLOAT_TYPE norm() const;

    [[nodiscard]] __device__ inline Vector3f normalized() const;

    [[nodiscard]] __device__ inline FLOAT_TYPE dot(const Vector3f& other) const;

    [[nodiscard]] __device__ inline Vector3f cross(const Vector3f& other) const;

    friend __host__ std::ostream& operator<<(std::ostream& out, Color vec);

    static __device__ inline Vector3f Zero();
    static __device__ inline Vector3f Random();

private:
    FLOAT_TYPE data[3];


};


using Color = Vector3f;

__device__ Vector3f operator*(FLOAT_TYPE s, const Vector3f& vec);


__device__ __host__ FLOAT_TYPE getRandom();


struct Ray {
public:
    /**
     * @brief Constructs a Ray from a Position and Direction
     * @param pos - Starting Position of the Ray
     * @param dir - Unit Direction where the Ray will be heading
     */
    __device__ Ray(Vector3f pos, Vector3f dir);

    /**
     * @brief Calculates where Ray will be in @a dist units of distance
     * @param dist - Distance Ray travels
     * @return Position Vector of new Ray location
     */
    __device__ Vector3f at(float dist) const;

    Vector3f pos; /** Position of the Ray **/
    Vector3f dir; /** Direction of the Ray **/
};


