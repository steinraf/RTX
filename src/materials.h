//
// Created by steinraf on 20/04/2022.
//

#pragma once

#include <utility>
#include "util.h"

/**
 * @brief Abstract Base class to describe all Materials
 */
class Material {
public:
    /**
     * @brief Describes how a ray should scatter from a specific material
     * @param[in] ray - Incoming Ray
     * @param[in] hit - Hit object describing the hit between the ray and the object
     * @return Outgoing ray and attenuation
     */
    virtual std::pair<Ray, double> scatter(const Ray &ray, const Hit &hit) const = 0;
};

/**
 * @brief Class describing a Lambertian Material
 */
class Lambertian : public Material{
public:
    /**
     * @brief Constructs a lambertian Material given a reflectivity
     * @param[in] reflectivity - A measure of how much light is reflected by the surface
     */
    Lambertian(double reflectivity);

    /**
     * @brief Scatters a ray that hits a surface randomly along a unit circle
     * @param[in] ray - Incoming Ray
     * @param[in] hit - Hit object describing the Ray-Shape intersection
     * @return returns the outgoing Ray and the attenuation (reflectivity)
     */
    std::pair<Ray, double> scatter(const Ray &ray, const Hit &hit) const override;
private:
    double reflectivity; /** A measure of how much light is reflected by the surface **/
};