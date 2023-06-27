#ifndef __RAY__
#define __RAY__

#include<vector3.h>

class ray {
public:
	ray() {};
	ray(const point& origin, const vector3& direction, double rtime = 0.0) : originPoint(origin), directionRay(direction), timeInterval(rtime) {};
	point origin() const;
	vector3 direction() const;
	point at(double t) const;
	double time() const;
private:
	point originPoint;
	vector3 directionRay;
	double timeInterval;
};

#endif