#include<aabb.h>
#include<vector3.h>
#include<cmath>

bool aabb::hit(ray& r, double t_min, double t_max) {
	for(int i = 0; i < 3; i++) {
		double invD = 1.0 / r.direction()[i];
		double t0 = (minimum[i] - r.origin()[i]) * invD;
		double t1 = (maximum[i] - r.origin()[i]) * invD;
		if(invD < 0) {
			std::swap(t0, t1);
		}
		t_min = std::max(t_min, t0);
		t_max = std::min(t_max, t1);
		if(t_max <= t_min) {
			return false;
		}
	}
	return true;
}

aabb surroundingBox(aabb box0, aabb box1) {
	point small = point(
		fmin(box0.min().x(), box1.min().x()),
		fmin(box0.min().y(), box1.min().y()),
		fmin(box0.min().z(), box1.min().z())
	);
	point big = point(
		fmax(box0.max().x(), box1.max().x()),
		fmax(box0.max().y(), box1.max().y()),
		fmax(box0.max().z(), box1.max().z())
	);

	return aabb(small, big);
}