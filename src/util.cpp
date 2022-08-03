//
// Created by steinraf on 16/04/2022.
//

#include "util.h"

Vector3f::Vector3f(float x, float y, float z)
        : data{x, y, z} {
}

Vector3f::Vector3f(std::array<float, 3> arr)
        : data(arr) {
}

Vector3f::Vector3f(const Vector3f &other)
        : data(other.data) {
}

Vector3f &Vector3f::operator=(const Vector3f &other) {
    return *this = Vector3f{other};
}

Vector3f Vector3f::Zero() {
    return Vector3f{0.0f, 0.0f, 0.0f};
}

Vector3f Vector3f::Random() {
    while(true){
        Vector3f vec{getRandom(), getRandom(), getRandom()};
        if(vec.norm() > 1)
            continue;

        return vec.normalized();
    }
}

float Vector3f::dot(const Vector3f &other) const {
    return std::transform_reduce(data.begin(), data.end(), other.data.begin(), 0.0f);
}

Vector3f Vector3f::cross(const Vector3f &other) const {
    return Vector3f{
        data[1]*other.data[2] - data[2]*other.data[1],
        data[2]*other.data[0] - data[0]*other.data[2],
        data[0]*other.data[1] - data[1]*other.data[0],
    };
}

Vector3f Vector3f::operator-(const Vector3f &other) const {
    std::array<float, 3> arr;
    std::transform(data.begin(), data.end(), other.data.begin(), arr.begin(), std::minus<>());
    return Vector3f{arr};
}

Vector3f Vector3f::operator+(const Vector3f &other) const {
    std::array<float, 3> arr;
    std::transform(data.begin(), data.end(), other.data.begin(), arr.begin(), std::plus<>());
    return Vector3f{arr};
}

Vector3f Vector3f::operator*(float scalar) const {
    std::array<float, 3> arr;
    std::transform(data.begin(), data.end(), arr.begin(), [scalar](float val) { return scalar * val; });
    return Vector3f{arr};
}

Vector3f operator*(float scalar, const Vector3f &vec) {
    return vec * scalar;
}

Vector3f Vector3f::operator/(float scalar) const {
    std::array<float, 3> arr;
    std::transform(data.begin(), data.end(), arr.begin(), [scalar](float val) { return val / scalar; });
    return Vector3f{arr};
}

float Vector3f::operator[](size_t idx) const{
    return data.at(idx);
}


float Vector3f::squaredNorm() const {
    return this->dot(*this);
}

float Vector3f::norm() const {
    return std::sqrt(this->squaredNorm());
}

Vector3f Vector3f::normalized() const {
    return *this/this->norm();
}


float degToRad(float deg) {
    return deg * M_PI / 180.0f;
}

Color::Color()
        : color(Vector3f::Zero()) {

}

Color::Color(float r, float g, float b)
        : color{r, g, b} {
    assert(std::clamp(r, 0.0f, 1.0f) == r && "R must be from 0.0 to 1.0");
    assert(std::clamp(g, 0.0f, 1.0f) == g && "R must be from 0.0 to 1.0");
    assert(std::clamp(b, 0.0f, 1.0f) == b && "R must be from 0.0 to 1.0");
}


Color Color::operator+(const Color &other) {
    return Color{color + other.color};
}

Color::Color(Vector3f c) :
        Color(c[0], c[1], c[2]) {

}

Color operator*(const Color &me, const float &k) {
    return Color{me.color * k};
}

Color operator*(const float &k, const Color &me) {
    return me * k;
}

Color Color::operator*(const Color &other) {
    return {color[0] * other.color[0], color[1] * other.color[1], color[2] * other.color[2]};
}

Camera::Camera(float pixelXResolution, float pixelYResolution, Vector3f pos,
               Vector3f lookAt, Vector3f up)
        : pos(pos), lookAt(lookAt), pixelXResolution(pixelXResolution), pixelYResolution(pixelYResolution) {

    float focusDist = 10.0f;//(lookAt-pos).norm();

    fov = degToRad(20);

    float h = std::tan(fov / 2);


    aspectRatio = static_cast<float>(pixelXResolution) / static_cast<float>(pixelYResolution);
    viewportHeight = 2.0f * h;
    viewportWidth = viewportHeight * aspectRatio;

    w = (pos - lookAt).normalized();
    u = up.cross(w).normalized();
    v = w.cross(u);

    horizontal = focusDist * viewportWidth * u;
    vertical = focusDist * viewportHeight * v;
    lowerLeft = pos - horizontal / 2.0f - vertical / 2.0f - focusDist * w;

    lensRadius = aperture / 2.0f;
}

std::ostream &operator<<(std::ostream &out, const Color &c) {
    float scale = 1.0f / c.subSamples;
    Color cNew{
            std::sqrt(scale * c.color[0]),
            std::sqrt(scale * c.color[1]),
            std::sqrt(scale * c.color[2])
    };
    out << static_cast<unsigned int>(std::clamp(cNew.color[0], 0.0f, 0.999f) * 256) << ' '
        << static_cast<unsigned int>(std::clamp(cNew.color[1], 0.0f, 0.999f) * 256) << ' '
        << static_cast<unsigned int>(std::clamp(cNew.color[2], 0.0f, 0.999f) * 256) << '\n';
    return out;
}

Color &Color::operator+=(const Color &other) {
    this->color = other.color + this->color;
    return *this;
}

float Color::asGray() {
    return (color[0] + color[1] + color[2]) / 3;
}


Ray Camera::getRay(float i, float j) const {
    // The rays are drawn starting from top left -> bottom right
    Vector3f rdm = getRandomInSphere() * lensRadius;
    Vector3f offset{rdm[0], rdm[1], 0};

    float localI = i / (pixelXResolution - 1);
    float localJ = 1 - j / (pixelYResolution - 1);

    Vector3f dir = lowerLeft + localI * horizontal + localJ * vertical - pos - offset;

    return Ray(pos + offset, dir);
}

Ray::Ray(Vector3f pos, Vector3f dir)
        : pos(pos), dir(dir.normalized()) {
}

Vector3f Ray::at(float dist) const {
    return pos + dist * dir;
}

float getRandom() {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(-1, 1);
    return dis(e);
}

inline Vector3f getRandomInSphere() {
    while (true) {
        Vector3f v{getRandom(), getRandom(), 0};
        if (v.norm() <= 1) return v;
    }
}


Timer::Timer()
        : start(std::chrono::high_resolution_clock::now()) {

}

float Timer::time() {
    std::chrono::duration<float> diff = std::chrono::high_resolution_clock::now() - start;
    return diff.count();
}

Benchmark::Benchmark(std::function<void(void)> f, unsigned int times) {
    std::vector<float> runtime(times);
    for (unsigned int i = 0; i < times; ++i) {
        Timer t;
        f();
        runtime[i] = t.time();
    }
    runtimes = runtime;
}

void Benchmark::plotHistogram() {

    std::sort(runtimes.begin(), runtimes.end());

    const float mean = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
    const float variance = std::accumulate(runtimes.begin(), runtimes.end(), 0.0,
                                           [mean](int a, int b) { return a + std::pow(b - mean, 2); }) /
                           runtimes.size();

    const float stddev = std::sqrt(variance);

    std::cout << "Mean:\t" << mean << '\n'
              << "Stddev:\t" << stddev << '\n';

    const unsigned int numBuckets = 10;
    std::array<int, numBuckets> arr;
    arr.fill(0);
    for (unsigned int i = 0; i < runtimes.size(); ++i) {
        int dist = (runtimes[i] - mean) / stddev + std::ceil(numBuckets / 2.0);
        ++arr[std::clamp(dist, 0, static_cast<int>(numBuckets - 1))];
    }

    for (auto num: arr)
        std::cout << std::string(num, '-') << '\n';
}


Hit::Hit(const Ray &ray, float dist, Vector3f normal)
        : ray(ray), dist(dist), normal(normal) {

}

Vector3f Hit::intersectPos() const {
    return ray.pos + ray.dir * dist;
}




