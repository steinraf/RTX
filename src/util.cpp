//
// Created by steinraf on 16/04/2022.
//

#include "util.h"

Color::Color(double r, double g, double b)
    : color({r, g, b}){
}

Color::Color()
    :color(Eigen::Vector3d::Zero()){

}

Color Color::operator+(const Color &other) {
    return Color{color + other.color};
}

Color::Color(Eigen::Vector3d c) :
    Color(c[0], c[1], c[2]){

}

Color operator*(const Color& me, const double &k){
    return Color{me.color * k};
}

Color operator*(const double &k, const Color& me){
    return me*k;
}

Camera::Camera(double pixelXResolution, double pixelYResolution, Eigen::Vector3d pos)
    : pixelXResolution(pixelXResolution), pixelYResolution(pixelYResolution), pos(pos){
    aspect_ratio = static_cast<double>(pixelXResolution)/static_cast<double>(pixelYResolution);
    viewportHeight = 2;
    viewportWidth = viewportHeight * aspect_ratio;
}

std::ostream& operator<<(std::ostream& out, const Color& c){
    double scale = 1.0/c.subSamples;
    Color cNew{
        sqrt(scale*c.color[0]),
        sqrt(scale*c.color[1]),
        sqrt(scale*c.color[2])
    };
    out << static_cast<unsigned int>(std::clamp(cNew.color[0], 0.0, 0.999)*256) << ' '
        << static_cast<unsigned int>(std::clamp(cNew.color[1], 0.0, 0.999)*256) << ' '
        << static_cast<unsigned int>(std::clamp(cNew.color[2], 0.0, 0.999)*256) << '\n';
    return out;
}

Color &Color::operator+=(const Color &other) {
    this->color = other.color + this->color;
    return *this;
}

double Color::asGray() {
    return (color[0] + color[1] + color[2])/3;
}

Ray Camera::getRay(double i, double j) const {
    // The rays are drawn starting from top left -> bottom right
    Eigen::Vector3d dir = {
            -viewportWidth/2 + i/(pixelXResolution-1)*viewportWidth,
            viewportHeight/2 - j/(pixelYResolution-1)*viewportHeight,
            -focal_length
    };
    return Ray(pos, dir.normalized());
}

Ray::Ray(Eigen::Vector3d pos, Eigen::Vector3d dir)
    :pos(pos), dir(dir){

}

double getRandom(){
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(-1, 1);
    return dis(e);
}
