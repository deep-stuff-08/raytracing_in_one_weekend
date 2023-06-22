#include<camera.h>
#include<cmath>

using namespace std;

camera::camera(vector3 eye, vector3 center, vector3 up, double vfov, double aspectRatio, double viewport, double focalLenght) {
	double fov = vfov * M_PI / 180.0;
	double h = tan(fov / 2);
	double vwportHeight = viewport * h;
	double vwportWidth = vwportHeight * aspectRatio;

	vector3 w = (eye - center).normalize();
	vector3 u = cross(up, w).normalize();
	vector3 v = cross(w, u);

	this->origin = eye;
	this->horizontal = u * vwportWidth;
	this->vertical = v * vwportHeight;
	this->lowerLeftConner = origin - horizontal/2 - vertical/2 - w;
}

ray camera::rayAt(double u, double v) {
	return ray(this->origin, this->lowerLeftConner + u * this->horizontal + v * this->vertical - origin);
}
