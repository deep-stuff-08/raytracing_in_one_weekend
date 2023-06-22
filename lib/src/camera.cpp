#include<camera.h>
#include<cmath>

using namespace std;

camera::camera(double vfov, double aspectRatio, double viewport, double focalLenght) {
	double fov = vfov * M_PI / 180.0;
	double h = tan(fov / 2);
	double vwportHeight = viewport * h;
	double vwportWidth = vwportHeight * aspectRatio;

	this->origin = vector3(0, 0, 0);
	this->horizontal = vector3(vwportWidth, 0, 0);
	this->vertical = vector3(0, vwportHeight, 0);
	this->lowerLeftConner = origin - horizontal/2 - vertical/2 - vector3(0, 0, focalLenght);
}

ray camera::rayAt(double u, double v) {
	return ray(this->origin, this->lowerLeftConner + u * this->horizontal + v * this->vertical - origin);
}
