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
    virtual std::pair<Ray, Color> scatter(const Ray &ray, const Hit &hit) const = 0;
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
    Lambertian(const Color &reflectivity);

    /**
     * @brief Scatters a ray that hits a surface randomly along a unit circle
     * @param[in] ray - Incoming Ray
     * @param[in] hit - Hit object describing the Ray-Shape intersection
     * @return returns the outgoing Ray and the attenuation (reflectivity)
     */
    std::pair<Ray, Color> scatter(const Ray &ray, const Hit &hit) const override;
private:
    Color reflectivity; /** A measure of how much light is reflected by the surface **/
};



/**
 * @brief Class describing a Metal-like reflective Material
 */
class Metal : public Material{
public:
    /**
     * @brief Constructs a Metal-like Material given a reflectivity
     * @param[in] reflectivity - A measure of how much light is reflected by the surface
     * @param[in]fuzziness - A measure of how much the reflection deviates from being perfect
     */
    Metal(const Color &reflectivity, double fuzziness = 0);

    /**
     * @brief Scatters a ray that hits a surface randomly along a unit circle
     * @param[in] ray - Incoming Ray
     * @param[in] hit - Hit object describing the Ray-Shape intersection
     * @return returns the outgoing Ray and the attenuation (reflectivity)
     */
    std::pair<Ray, Color> scatter(const Ray &ray, const Hit &hit) const override;
private:
    Color reflectivity; /** A measure of how much light is reflected by the surface **/
    double fuzziness; /** A measure of the magnitude of the perturbation from the perfect reflextion **/
};

/**
 * @brief Class describing a transparent Dielectric Material
 */
class Dielectric : public Material{
public:
    /**
     * @brief Constructs a Dielectric Material, like Glass
     * @param[in] reflectivity - A measure of what wavelength the material absorbs
     * @param[in] refractiveIndex - A measure of how much Rays are bent on changing Mediums
     */
    Dielectric(const Color &reflectivity, double refractiveIndex);

    /**
     * @brief Scatters a ray that interacts with the Surface of the Dielectric Material
     * @param[in] ray - Incoming ray
     * @param[in] hit - Hit object describing the interaction at the Surface
     * @return returns the outgoing Ray and Attenuation
     */
    std::pair<Ray, Color> scatter(const Ray &ray, const Hit &hit) const override;

private:
    Color reflectivity; /** A measure of which wavelengths are reflected by the surface **/
    double refractiveIndex; /** A measure of how much Rays are bent on switching mediums **/
};