#include<basiccamera.h>

using namespace std;

camera::camera(double aspectRatio, double viewport, double focalLenght) {
	double vwportHeight = viewport;
	double vwportWidth = viewport * aspectRatio;

	this->origin = vector3(0, 0, 0);
	this->horizontal = vector3(vwportWidth, 0, 0);
	this->vertical = vector3(0, vwportHeight, 0);
	this->lowerLeftConner = origin - horizontal/2 - vertical/2 - vector3(0, 0, focalLenght);
}

ray camera::rayAt(double u, double v) {
	return ray(this->origin, this->lowerLeftConner + u * this->horizontal + v * this->vertical - origin);
}
