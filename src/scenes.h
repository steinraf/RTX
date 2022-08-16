//
// Created by steinraf on 10/08/22.
//

#pragma once

#include "sphere.h"
#include "material.h"
#include "hitable_list.h"

__device__ void create_spheres(Hittable **d_list, Hittable **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new Sphere(Vector3f(0, -100.5, -1), 100,
                               new lambertian(Vector3f(0.8, 0.8, 0.0)));
        d_list[1] = new Sphere(Vector3f(0, 0, -1), 0.1,
                               new lambertian(Vector3f(0.1, 0.2, 0.5)));
        d_list[2] = new Sphere(Vector3f(4, 0.7, -10), 0.5,
                               new dielectric(1.5));
//        d_list[3] = new Sphere(Vector3f(2, 0.7, -10), 0.5,
//                               new dielectric(1.5));
        d_list[3] = new Sphere(Vector3f(1, 0, -1), 0.3,
                               new metal(Vector3f(0.8, 0.6, 0.2), 0.0));
        d_list[4] = new Sphere(Vector3f(1, 0, -1), 0.2,
                               new dielectric(1.5));
        d_list[5] = new Sphere(Vector3f(-1, 0, -1), 0.5,
                               new dielectric(1.5));
        d_list[6] = new Sphere(Vector3f(-1, 0, -1), -0.45,
                               new dielectric(1.5));
        *d_world = new hitable_list(d_list, 7);
        Vector3f lookfrom(-3,1,5);
        Vector3f lookat(0, 0, -1);
        float dist_to_focus = (lookfrom - lookat).norm();
        float aperture = 0.0f;
        *d_camera = new Camera(lookfrom,
                               lookat,
                               Vector3f(0, 1, 0),
                               20.0,
                               float(nx) / float(ny),
                               aperture,
                               dist_to_focus);
    }
}

__device__ void delete_spheres(Hittable **d_list, Hittable **d_world, Camera **d_camera){
    for (int i = 0; i < 7; i++) {
        delete ((Sphere *) d_list[i])->mat_ptr;
        delete d_list[i];
    }

    delete *d_world;
    delete *d_camera;
}


__device__ void create_room(Hittable **d_list, Hittable **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        //left
        d_list[0] = new Sphere(Vector3f(-100.5f, 0, 0), 100,
                               new metal(Vector3f(0.0f, 1.0f, 0.0f), 0.0f));
        //right
        d_list[1] = new Sphere(Vector3f( 100.5f, 0, 0), 100,
                               new metal(Vector3f(1.0f, 0.0f, 0.0f), 0.0f));
        //bottom
        d_list[2] = new Sphere(Vector3f(0, -100.5f, 0), 100,
                               new metal(Vector3f(1.0f, 1.0f, 1.0f), 0.5f));
        //top
        d_list[3] = new Sphere(Vector3f(0,  100.5f, 0), 100,
                               new metal(Vector3f(1.0f, 1.0f, 1.0f), 0.5f));

//        back
        d_list[4] = new Sphere(Vector3f(0, 0, -200.5f), 100,
                               new metal(Vector3f(1.0f, 1.0f, 1.0f), 1.0f));


        *d_world = new hitable_list(d_list, 5);


        Vector3f lookfrom(0,0,1);
        Vector3f lookat(0, 0, 0);
        Vector3f up(0, 1, 0);
        float dist_to_focus = (lookfrom - lookat).norm();
        float aperture = 0.0f;
        *d_camera = new Camera(lookfrom,
                               lookat,
                               up,
                               40.0,
                               float(nx) / float(ny),
                               aperture,
                               dist_to_focus);
    }
}

__device__ void delete_room(Hittable **d_list, Hittable **d_world, Camera **d_camera){
    for (int i = 0; i < 5; i++) {
        delete ((Sphere *) d_list[i])->mat_ptr;
        delete d_list[i];
    }

    delete *d_world;
    delete *d_camera;
}


__device__ void create_cover0(Hittable **d_list, Hittable **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state){
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    curandState localState = rand_state[0];

    d_list[0] = new Sphere(Vector3f(0, -1000, 0), 1000,
                           new lambertian(Vector3f(0.5, 0.5, 0.5)));

    size_t counter = 0;

    auto rand = [&localState](){
        return curand_uniform(&localState);
    };

    for (int x = -11; x < 11; ++x) {
        for (int z = -11; z < 11; z++) {
            float r = rand();
            Vector3f pos = {x + 0.9f*rand(), 0.2, z + 0.9f*rand()};
            if(r < 0.8f){
                d_list[++counter] = new Sphere(pos, 0.2,
                                             new lambertian(Vector3f(rand()*rand(), rand()*rand(), rand()*rand()))
                        );
            } else if(r < 0.95f){
                d_list[++counter] = new Sphere(pos, 0.2,
                                             new metal(Vector3f(rand()/2+0.5, rand()/2+0.5, rand()/2+0.5), rand()/2.0f)
                );
            } else {
                d_list[++counter] = new Sphere(pos, 0.2,
                                             new dielectric(1.5)
                );
            }
        }
    }

    d_list[++counter] = new Sphere(Vector3f(0, 1, 0), 1.0,
                                   new dielectric(1.5));

    d_list[++counter] = new Sphere(Vector3f(-4, 1, 0), 1.0,
                                   new lambertian(Vector3f(0.4f, 0.2f, 0.1f)));


    d_list[++counter] = new Sphere(Vector3f(4, 1, 0), 1.0,
                                   new metal(Vector3f(0.7, 0.6, 0.5), 0.0));



    *d_world = new hitable_list(d_list, counter+1);


    Vector3f lookfrom{13,2,3};
    Vector3f lookat(0, 0, 0);
    float dist_to_focus = 10;//(lookfrom - lookat).norm();
    *d_camera = new Camera(lookfrom,
                           lookat,
                           Vector3f(0, 1, 0),
                           20.0f,
                           float(nx) / float(ny),
                           0.1f,
                           dist_to_focus);

}

__device__ void delete_cover0(Hittable **d_list, Hittable **d_world, Camera **d_camera){
    for (int i = 0; i < 488; i++) {
        delete ((Sphere *) d_list[i])->mat_ptr;
        delete d_list[i];
    }

    delete *d_world;
    delete *d_camera;
}