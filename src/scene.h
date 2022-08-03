//
// Created by steinraf on 16/04/2022.
//

#pragma once

#include "util.h"
#include "shapes.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <atomic>

/**
 * @brief Class that handles all objects in a scene and allows it to be rendered by casting Rays
 */
class Scene {
public:
    /**
     * @brief Constructs a scene which then can be used to cast Rays
     * @param[in] width - x dimension of Scene
     * @param[in] height  - y dimension of Scene
     * @param[in] subSamples - Amount of Rays cast per Pixel of the Scene
     * @param[in] maxRayDepth - Maximum Amount of times a Ray can reflect from surfaces
     */
    __device__ Scene(unsigned int width, unsigned int height, unsigned int subSamples = 100, unsigned int maxRayDepth = 50);
    __device__ ~Scene();
    /**
     * @brief Renders an image and creates a ppm image at the location of the Path
     */
    __device__ void render();

    /**
     * @brief Adds Shapes to the scene
     * @param[in] shape - Shape to be added to the Scene
     */
    __device__ void addShape(Shape * shape);

private:

    /**
     * @brief Finds the cloest Point where the Ray intersects a Shape in the Scene
     * @param[in] r - Ray that is used
     * @return returns the closest Shape and the distance to it
     */
    __device__ std::pair<Shape *, double> getClosestIntersect(const Ray &r) const;


    /**
     * @brief Will find the Color for a Ray that didn't hit a target
     * @param ray - Ray to be considered
     * @return Color of the background for a given Ray
     */
    __device__ Color getBackground(const Ray &ray) const;

    /**
     * @brief Finds the color for a Ray that is cast
     * @param[in] ray - Ray that is cast
     * @param[in] maxRayDepth - Maximum recursion depth
     * @return Color of the Ray after maximum @a maxRayDepth bounces
     */
    __device__ Color castRay(const Ray &ray, int maxRayDepth) const;


    unsigned int width; /** Width of the Scene **/
    unsigned int height; /** Height of the Scene **/

    Camera camera; /** Camera object that casts out the Rays **/

    const int subSamples; /** Amount of Rays cast per Pixel **/
    const int maxRayDepth; /** Maximum amount of times a Ray can reflect from a Surface **/

    const std::string base_path = std::filesystem::path(__FILE__).parent_path().parent_path();
    const std::string path = base_path + "/data/image.ppm"; /** Path where image will be saved **/

    std::vector<Shape*> shapes; /** Vector to save all Shapes **/
};


