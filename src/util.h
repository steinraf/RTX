//
// Created by steinraf on 16/04/2022.
//

#pragma once
#include "eigen3/Eigen/Dense"
#include <memory>
#include <random>
//#include "shapes.h"

class Color{
public:
    Color();
    Color(double r, double g, double b);
    Color(Eigen::Vector3d c);
    Color operator+(const Color& other);
    Color& operator+=(const Color& other);
    friend Color operator*(const Color& me, const double &k);
    friend Color operator*(const double &k, const Color& me);
    friend std::ostream& operator<<(std::ostream& out, const Color& c);

    double asGray();

    int subSamples;
private:
    Eigen::Vector3d color;
};


class Ray{
public:
    Ray(Eigen::Vector3d pos, Eigen::Vector3d dir);

    Eigen::Vector3d pos;
    Eigen::Vector3d dir;

private:

};

class Camera{
public:
    Camera(double pixelXResolution, double pixelYResolution, Eigen::Vector3d pos={0, 0, 0});
    Ray getRay(double i, double j) const;
private:
    Eigen::Vector3d pos;

    double focal_length = 1;
    double pixelXResolution;
    double pixelYResolution;
    double aspect_ratio;
    double viewportHeight;
    double viewportWidth;
};

double getRandom();




