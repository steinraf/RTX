#pragma once

#include <curand_kernel.h>
#include "Ray.h"

__device__ Vector3f random_in_unit_disk(curandState *local_rand_state) {
    Vector3f p;
    do {
        p = 2.0f * Vector3f(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - Vector3f(1, 1, 0);
    } while (p.squaredNorm() >= 1.0f);
    return p;
}

class Camera {
public:
    __device__ Camera(Vector3f lookfrom, Vector3f lookat, Vector3f vup, float vfov, float aspect, float aperture,
                      float focus_dist) { // vfov is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        float theta = vfov * ((float) M_PI) / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(vup.cross(w));
        v = w.cross(u);
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;
    }

    __device__ Ray get_ray(float s, float t, curandState *local_rand_state) {
        Vector3f rd = lens_radius * random_in_unit_disk(local_rand_state);
        Vector3f offset = u * rd[0] + v * rd[1];
        return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    Vector3f origin;
    Vector3f lower_left_corner;
    Vector3f horizontal;
    Vector3f vertical;
    Vector3f u, v, w;
    float lens_radius;
};

