#ifndef HITABLEH
#define HITABLEH

#include "Ray.h"

class material;

struct hit_record
{
    float t;
    Vector3f p;
    Vector3f normal;
    material *mat_ptr;
};

class Hittable  {
    public:
        __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif
