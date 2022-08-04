//
// Created by steinraf on 04/08/22.
//

#pragma once


/**
 * @brief Abstract Base class to describe all Shapes that make up the scene
 */
class Shape {
public:

    /**
     * @brief Adds a Material pointer to the Shape Base
     * @param mat
     */
    __device__ Shape(Material *mat);
    __device__ virtual ~Shape() noexcept;

    /**
     * @brief Returns the distance a ray needs to travel to intersect the Shape
     * @param[in] ray - Cast Ray
     * @return Distance to Shape
     */
    __device__ virtual float findIntersect(const Ray &ray) const = 0;

    /**
     * @brief Given a position inside the Shape (Used in combination with @b findIntersect ) returns the Surface normal facing outwards
     * @param[in] pos - Position close to the Shape Surface
     * @return normalized Surface normal closest to pos
     */
    __device__ virtual Vector3f getNormal(const Vector3f &pos) const = 0;

    /**
     * @brief Calculates Hit object for a ray intersection at distance @a intersect
     * @param[in] ray - Ray that will intersect Shape
     * @param[in] intersect - Distance until Ray intersects Shape
     * @return Hit object describing the intersection
     */
    [[nodiscard]] __device__ virtual Hit calculateHit(const Ray &ray, float intersect) = 0;

    /**
     * @brief Conventient overload for @c calculateHit( const @a Ray &ray, @a float intersect)
     * @param[in] ray - Ray that will intersect Shape
     * @return Hit object describing the intersection
     */
    [[nodiscard, maybe_unused]] __device__ virtual Hit calculateHit(const Ray &ray){
        return this->calculateHit(ray, findIntersect(ray));
    }

    /**
     * @brief Describes how a Shape scatters an incoming ray, defaults to material property
     * @param[in] ray - Ray that is on a collision course with given Shape
     * @param[in] hit - Hit object describing the Ray-Shape Intersection
     * @return Ray after scattering off the Surface
     */
    [[nodiscard]] virtual __device__ std::pair<Ray, Color> scatter(const Ray &ray, const Hit &hit) const {
        assert(material && "Undefined Material of Shape");
        return material->scatter(ray, hit);
    };

private:
    Material *material; /** Material that is associated with Shape **/


};

/**
 * @brief Shape describing a Sphere
 */
class Sphere : public Shape {
public:
    /**
     * @brief Creates a Sphere
     * @param[in] pos - Center of the Sphere
     * @param[in] r - Radius of the Sphere
     * @param[in] mat - Material of Sphere
     */
    __device__ Sphere(Vector3f pos, float r, Material *mat);

    /**
     * @brief Finds the intersection of a Ray with a Sphere
     * @param[in] ray - Ray to be considered
     * @return distance of closest intersection
     */
    __device__ float findIntersect(const Ray &ray) const override;

    /**
     * @brief Finds the normal for the Intersection
     * @param[in] pos - Position where Sphere is penetrated
     * @return Normalized Sphere Normal vector
     */
    __device__ Vector3f getNormal(const Vector3f &pos) const override;

    /**
     * @brief Calculates the Hit object for a given Ray
     * @param ray - Ray that intersects the Sphere
     * @param intersect - Distance where Sphere is intersected
     * @return Hit object of Intersection
     */
    [[nodiscard]] __device__ Hit calculateHit(const Ray &ray, float intersect) override;


private:
    Vector3f center; /** Center of the Sphere **/
    float radius; /** Radius of the Sphere **/
};