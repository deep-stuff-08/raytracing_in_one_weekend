#ifndef __RAY__
#define __RAY__

#include<vector3.h>

class ray {
public:
	ray() {};
	ray(const point& origin, const vector3& direction) : originPoint(origin), directionRay(direction) {};
	point origin() const;
	vector3 direction() const;
	point at(double t) const;
private:
	point originPoint;
	vector3 directionRay;
};

#endif