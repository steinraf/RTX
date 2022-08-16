#pragma once

struct hit_record;

#include "Ray.h"
#include "Hittable.h"


__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const Vector3f &v, const Vector3f &n, float ni_over_nt, Vector3f &refracted) {
    Vector3f uv = unit_vector(v);
    float dt = uv.dot(n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    } else
        return false;
}

#define RANDVEC3 Vector3f(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ Vector3f random_in_unit_sphere(curandState *local_rand_state) {
    Vector3f p;
    do {
        p = 2.0f * RANDVEC3 - Vector3f(1, 1, 1);
    } while (p.squaredNorm() >= 1.0f);
    return p;
}

__device__ Vector3f reflect(const Vector3f &v, const Vector3f &n) {
    return v - 2.0f * v.dot(n) * n;
}

class material {
public:
    __device__ virtual bool scatter(const Ray &r_in, const hit_record &rec, Vector3f &attenuation, Ray &scattered,
                                    curandState *local_rand_state) const = 0;
};

class lambertian : public material {
public:
    __device__ lambertian(const Vector3f &a) : albedo(a) {}

    __device__ virtual bool scatter(const Ray &r_in, const hit_record &rec, Vector3f &attenuation, Ray &scattered,
                                    curandState *local_rand_state) const {
        Vector3f target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = Ray(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    }

    Vector3f albedo;
};

class metal : public material {
public:
    __device__ metal(const Vector3f &a, float fuzziness) : albedo(a) { if (fuzziness < 1) fuzz = fuzziness; else fuzz = 1; }

    __device__ virtual bool scatter(const Ray &r_in, const hit_record &rec, Vector3f &attenuation, Ray &scattered,
                                    curandState *local_rand_state) const {
        Vector3f reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (scattered.direction().dot(rec.normal) > 0.0f);
    }

    Vector3f albedo;
    float fuzz;
};

class dielectric : public material {
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}

    __device__ virtual bool scatter(const Ray &r_in,
                                    const hit_record &rec,
                                    Vector3f &attenuation,
                                    Ray &scattered,
                                    curandState *local_rand_state) const {
        Vector3f outward_normal;
        Vector3f reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = Vector3f(1.0, 1.0, 1.0);
        Vector3f refracted;
        float reflect_prob;
        float cosine;
        if (r_in.direction().dot(rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = r_in.direction().dot(rec.normal) / r_in.direction().norm();
            cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
        } else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -r_in.direction().dot(rec.normal) / r_in.direction().norm();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = Ray(rec.p, reflected);
        else
            scattered = Ray(rec.p, refracted);
        return true;
    }

    float ref_idx;
};
