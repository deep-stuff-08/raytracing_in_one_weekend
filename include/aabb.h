#ifndef __AABB__
#define __AABB__

#include<vector3.h>
#include<ray.h>

class aabb {
private:
	point minimum;
	point maximum;
public:
	aabb() {}
	aabb(point min, point max): minimum(min), maximum(max) {}
	point min() { return minimum; }
	point max() { return maximum; }
	bool hit(const ray &r, double t_min, double t_max) const;
};

aabb surroundingBox(aabb box0, aabb box1);

#endif