//
// Created by steinraf on 16/04/2022.
//

#pragma once

#include <memory>
#include <random>
#include <chrono>
#include <iostream>
#include <functional>
#include <cassert>

/**
 * @brief Replacement of Eigen::Vector3f
 */
class Vector3f{
public:
    Vector3f(){}
    Vector3f(float x, float y, float z);
    Vector3f(std::array<float, 3> arr);
    ~Vector3f() = default;
    Vector3f(const Vector3f& other);
    Vector3f(Vector3f&& other) noexcept = default;
    Vector3f& operator=(const Vector3f& other);
    Vector3f& operator=(Vector3f&& other) = default;

    [[nodiscard]] Vector3f operator-(const Vector3f& other) const;

    [[nodiscard]] Vector3f operator+(const Vector3f& other) const;

    friend Vector3f operator*(float s, const Vector3f& vec);
    [[nodiscard]] Vector3f operator*(float scalar) const;

    [[nodiscard]] Vector3f operator/(float scalar) const;

    [[nodiscard]] float operator[](size_t idx) const;

    [[nodiscard]] float squaredNorm() const;

    [[nodiscard]] float norm() const;

    [[nodiscard]] Vector3f normalized() const;

    [[nodiscard]] float dot(const Vector3f& other) const;

    [[nodiscard]] Vector3f cross(const Vector3f& other) const;

    static Vector3f Zero();
    static Vector3f Random();

private:
    std::array<float, 3> data;

};

/**
 * @brief Provides overload for scalar vector multiplication
 * @param scalar float to scale
 * @param vec vector to be scaled
 * @return vec * scalar
 */
Vector3f operator*(float scalar, const Vector3f &vec);

/**
 * @brief Defines the conversion between Degrees and Radians
 * @param[in] deg - Degrees
 * @return Radians
 */
inline float degToRad(float deg);

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
    Color(float r, float g, float b);

    /**
     * @brief Initializes a Color from a Vector
     * @param[in] c - Color as Vector
     */
    Color(Vector3f c);

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
    friend Color operator*(const Color &me, const float &k);

    /**
     * @brief Defines the scaling of a Color by a constant
     * @param[in] k - amount of Scaling
     * @param[in] me - Color to be scaled
     * @return Color value scaled by @a k
     */
    friend Color operator*(const float &k, const Color &me);

    /**
     * @brief Defines the product of two Colors
     * @param[in] other - Other Color
     * @return elementwise product of Colors
     */
    Color operator*(const Color &other);

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
    float asGray();

    int subSamples; /** Subsample count needed for Gamma Correction **/
private:
    Vector3f color; /** Internal Color representation **/
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
    Ray(Vector3f pos, Vector3f dir);

    /**
     * @brief Calculates where Ray will be in @a dist units of distance
     * @param dist - Distance Ray travels
     * @return Position Vector of new Ray location
     */
    Vector3f at(float dist) const;

    Vector3f pos; /** Position of the Ray **/
    Vector3f dir; /** Direction of the Ray **/
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
    Hit(const Ray &ray, float dist, Vector3f normal);

    /**
     * @brief Gets the Ray-Shape intersection position
     * @return Position of Ray in @a dist units
     */
    [[nodiscard]] Vector3f intersectPos() const;

    Ray ray; /** Ray object before intersection **/
    float dist; /** Distance from ray position to intersection **/
    Vector3f normal; /** Normal Vector of collision **/
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
    Camera(float pixelXResolution, float pixelYResolution,
           Vector3f pos = {13.0f,2.0f,3.0f}, Vector3f lookAt = {0.0f, 0.0f, 0.0f},
           Vector3f up = {0.0f, 1.0f, 0.0f});

    /**
     * @brief Returns the Ray for a given position in the Camera View
     * @param[in] i - x index of Ray
     * @param[in] j - y index of Ray
     * @return Ray getting sent out of the Camera
     */
    Ray getRay(float i, float j) const;

private:

    Vector3f pos; /** Camera Position **/

    Vector3f lookAt; /** Where Camera is facing **/

    Vector3f u; /** Local x coordinate **/
    Vector3f v; /** Local y coordinate **/
    Vector3f w; /** Local z coordinate **/

    float pixelXResolution; /** Amount of Pixels in x direction **/
    float pixelYResolution; /** Amount of Pixels in y direction **/

    float aspectRatio; /** Ratio of Width/Height of the Viewport **/
    float viewportHeight; /** Height of the Viewport **/
    float viewportWidth; /** Width of the Viewport **/

    float aperture = 0.1f; /** Size of the Camera Hole **/
    float lensRadius; /** Size of the Lens **/
    float fov; /** Camera field of view **/

    Vector3f horizontal; /** Local x-coordinates **/
    Vector3f vertical; /** Local y-coordinates **/
    Vector3f lowerLeft; /** World bottomLeft corner **/

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
    float time();

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
    std::vector<float> runtimes; /** Storage of individual runtimes **/
};

/**
 * @brief Returns a random number
 * @return Returns a random number from -1.0f to 1.0f
 */
float getRandom();

/**
 * @brief Returns a random point in the x-y unit sphere
 * @return Point inside the unit sphere
 */
inline Vector3f getRandomInSphere();


