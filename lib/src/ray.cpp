#include<ray.h>

point ray::origin() const {
	return originPoint;
}

vector3 ray::direction() const {
	return directionRay;
}

point ray::at(double t) const {
	return originPoint + directionRay * t;
}
