#ifndef __RAY__
#define __RAY__

#include<vector3.cuh>

class ray {
public:
	__device__ ray() {};
	__device__ ray(const point& origin, const vector3& direction, double rtime = 0.0) : originPoint(origin), directionRay(direction) {};
	__device__ point origin() const;
	__device__ vector3 direction() const;
	__device__ point at(double t) const;
private:
	point originPoint;
	vector3 directionRay;
};

__device__ point ray::origin() const {
	return originPoint;
}

__device__ vector3 ray::direction() const {
	return directionRay;
}

__device__ point ray::at(double t) const {
	return originPoint + directionRay * t;
}

#endif