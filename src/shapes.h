//
// Created by steinraf on 16/04/2022.
//

#pragma once
#include "util.h"


class Shape{
public:
    virtual double findIntersect(const Ray& ray) const = 0;
    virtual Ray reflect(const Ray &ray) const = 0;

    double reflectivity = 0.5;
private:


};

class Sphere : public Shape{
public:
    Sphere(Eigen::Vector3d pos, double r, double reflectivity=0.5);
    double findIntersect(const Ray& ray) const override;
    Ray reflect(const Ray &ray) const override;

    double reflectivity;

private:
    Eigen::Vector3d pos;
    double radius;


};


