//
// Created by steinraf on 16/04/2022.
//

#pragma once

#include "util.h"
#include "shapes.h"
#include <fstream>
#include <iostream>
#include <filesystem>


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
    Scene(unsigned int width, unsigned int height, unsigned int subSamples = 100, unsigned int maxRayDepth = 50);

    /**
     * @brief Renders an image and creates a ppm image at the location of the Path
     */
    void render();

    /**
     * @brief Adds Shapes to the scene
     * @param[in] shape - Shape to be added to the Scene
     */
    void addShape(std::shared_ptr<Shape> shape);

private:

    /**
     * @brief Finds the cloest Point where the Ray intersects a Shape in the Scene
     * @param[in] r - Ray that is used
     * @return returns the closest Shape and the distance to it
     */
    std::pair<std::shared_ptr<Shape>, double> getClosestIntersect(const Ray &r) const;


    /**
     * @brief Will find the Color for a Ray that didn't hit a target
     * @param ray - Ray to be considered
     * @return Color of the background for a given Ray
     */
    Color getBackground(const Ray &ray) const;

    /**
     * @brief Finds the color for a Ray that is cast
     * @param[in] ray - Ray that is cast
     * @param[in] maxRayDepth - Maximum recursion depth
     * @return Color of the Ray after maximum @a maxRayDepth bounces
     */
    Color castRay(const Ray &ray, int maxRayDepth) const;


    unsigned int width; /** Width of the Scene **/
    unsigned int height; /** Height of the Scene **/

    Camera camera; /** Camera object that casts out the Rays **/

    const int subSamples; /** Amount of Rays cast per Pixel **/
    const int maxRayDepth; /** Maximum amount of times a Ray can reflect from a Surface **/

    const std::string base_path = std::filesystem::path(__FILE__).parent_path().parent_path();
    const std::string path = base_path + "/data/image.ppm"; /** Path where image will be saved **/

    std::vector<std::shared_ptr<Shape>> shapes; /** Vector to save all Shapes **/
};


