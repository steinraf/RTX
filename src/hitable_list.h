#ifndef HITABLELISTH
#define HITABLELISTH

#include "Hittable.h"

class hitable_list: public Hittable  {
    public:
        __device__ hitable_list() {}
        __device__ hitable_list(Hittable **l, int n) {list = l; list_size = n; }
        __device__ virtual bool hit(const Ray& r, float tmin, float tmax, hit_record& rec) const;
        Hittable **list;
        int list_size;
};

__device__ bool hitable_list::hit(const Ray& r, float t_min, float t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
}

#endif
