#ifndef __AABB__
#define __AABB__

#include<vector3.h>
#include<ray.h>

class aabb {
private:
	point minimum;
	point maximum;
public:
	aabb(point min, point max): minimum(min), maximum(max) {}
	point min() { return minimum; }
	point max() { return maximum; }
	bool hit(ray &r, double t_min, double t_max);
};

#endif