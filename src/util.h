//
// Created by steinraf on 16/04/2022.
//

#pragma once

#include "eigen3/Eigen/Dense"
#include <memory>
#include <random>
#include <chrono>
#include <iostream>

/**
 * @brief Class describing a Color value
 */
class Color {
public:
    /**
     * @brief Initializes a Color as Black
     */
    Color();

    /**
     * @brief Initializes an RGB Color
     * @param[in] r - Red Value in [0.0, 1.0]
     * @param[in] g - Green Value in [0.0, 1.0]
     * @param[in] b - Blue Value in [0.0, 1.0]
     */
    Color(double r, double g, double b);

    /**
     * @brief Initializes a Color from a Vector
     * @param[in] c - Color as Vector
     */
    Color(Eigen::Vector3d c);

    /**
     * @brief Defines addition between Colors
     * @param[in] other - Color to be added
     * @return Added Colors
     */
    Color operator+(const Color &other);

    /**
     * @brief Defines addition between Colors
     * @param[in] other - Color to be added
     * @return Reference to first color
     */
    Color &operator+=(const Color &other);

    /**
     * @brief Defines the scaling of a Color by a constant
     * @param[in] me - Color to be scaled
     * @param[in] k - amount of Scaling
     * @return Color value scaled by @a k
     */
    friend Color operator*(const Color &me, const double &k);

    /**
     * @brief Defines the scaling of a Color by a constant
     * @param[in] k - amount of Scaling
     * @param[in] me - Color to be scaled
     * @return Color value scaled by @a k
     */
    friend Color operator*(const double &k, const Color &me);

    /**
     * @brief Defines the printing operator for a Color
     * @param[in] out - Stream where the Color will be written to
     * @param c - Color to be written
     * @return Reference to the Stream
     */
    friend std::ostream &operator<<(std::ostream &out, const Color &c);

    /**
     * @brief Returns the Gray value of the number
     * @return Grayscale Value of the Color in [0.0, 1.0]
     */
    double asGray();

    int subSamples; /** Subsample count needed for Gamma Correction **/
private:
    Eigen::Vector3d color; /** Internal Color representation **/
};

/**
 * @brief Struct describing a Ray of Light
 */
struct Ray {
public:
    /**
     * @brief Constructs a Ray from a Position and Direction
     * @param pos - Starting Position of the Ray
     * @param dir - Unit Direction where the Ray will be heading
     */
    Ray(Eigen::Vector3d pos, Eigen::Vector3d dir);

    /**
     * @brief Calculates where Ray will be in @a dist units of distance
     * @param dist - Distance Ray travels
     * @return Position Vector of new Ray location
     */
    Eigen::Vector3d at(double dist) const;

    Eigen::Vector3d pos; /** Position of the Ray **/
    Eigen::Vector3d dir; /** Direction of the Ray **/
};


/**
 * @brief Saves the Information of the closest Ray-Shape intersection
 */
struct Hit {
public:
    /**
     * @brief Constructs a hit
     * @param[in] ray - Ray that hit the Shape
     * @param[in] dist - Distance of intersection, starting from ray position
     * @param[in] normal - Normal of intersection
     */
    Hit(const Ray &ray, double dist, Eigen::Vector3d normal);

    /**
     * @brief Gets the Ray-Shape intersection position
     * @return Position of Ray in @a dist units
     */
    [[nodiscard]] inline Eigen::Vector3d intersectPos() const;

    Ray ray; /** Ray object before intersection **/
    double dist; /** Distance from ray position to intersection **/
    Eigen::Vector3d normal; /** Normal Vector of collision **/
};

/**
 * @brief Camera Object that sends out the Rays
 */
class Camera {
public:
    /**
     * @brief Creates a Camera
     * @param[in] pixelXResolution - Amount of Pixels in the x direction
     * @param[in] pixelYResolution - Amount of Pixels in the y direction
     * @param[in] pos - Position of the Camera
     */
    Camera(double pixelXResolution, double pixelYResolution, Eigen::Vector3d pos = {0, 0, 0});

    /**
     * @brief Returns the Ray for a given position in the Camera View
     * @param[in] i - x index of Ray
     * @param[in] j - y index of Ray
     * @return Ray getting sent out of the Camera
     */
    Ray getRay(double i, double j) const;

private:

    Eigen::Vector3d pos; /** Camera Position **/

    double pixelXResolution; /** Amount of Pixels in x direction **/
    double pixelYResolution; /** Amount of Pixels in y direction **/


    double aspectRatio; /** Ratio of Width/Height of the Viewport **/
    double viewportHeight; /** Height of the Viewport **/
    double viewportWidth; /** Width of the Viewport **/

    double focalLength = 1; /** Focal Length of the Camera **/
};

/**
 * @brief Class used for timing
 */
class Timer {
public:
    /**
     * @brief Initializes a timer
     */
    Timer();

    /**
     * @brief Measures time since initialization
     * @return Returns seconds since initialization
     */
    double time();

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start; /** Starting time **/
};

/**
 * @brief Class used for benchmarking functions
 */
class Benchmark {
public:
    /**
     * @brief Initializes and runs a Benchmark
     * @param[in] f - Function to be timed
     * @param[in] times - How many times the function should be averaged over
     */
    Benchmark(std::function<void(void)> f, unsigned int times);

    /**
     * @brief Prints a histogram of the run distributions to the console
     */
    void plotHistogram();

private:
    std::vector<double> runtimes; /** Storage of individual runtimes **/
};

/**
 * @brief Returns a random number
 * @return Returns a random number from -1.0 to 1.0
 */
double getRandom();
