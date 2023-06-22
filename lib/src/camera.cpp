#include<camera.h>
#include<cmath>

using namespace std;

camera::camera(vector3 eye, vector3 center, vector3 up, double vfov, double aspectRatio, double aperature, double focalDist) {
	double fov = vfov * M_PI / 180.0;
	double h = tan(fov / 2);
	double vwportHeight = 2 * h;
	double vwportWidth = vwportHeight * aspectRatio;

	this->w = (eye - center).normalize();
	this->u = cross(up, w).normalize();
	this->v = cross(w, u);

	this->origin = eye;
	this->horizontal = u * vwportWidth * focalDist;
	this->vertical = v * vwportHeight * focalDist;
	this->lowerLeftConner = origin - horizontal/2 - vertical/2 - focalDist * w;

	this->lensRadius = aperature / 2.0;
}

ray camera::rayAt(double s, double t) {
	vector3 rd = lensRadius * random_in_unit_disk();
	vector3 offset = this->u * rd.x() + this->v * rd.y();
	return ray(this->origin + offset, this->lowerLeftConner + s * this->horizontal + t * this->vertical - origin - offset);
}
