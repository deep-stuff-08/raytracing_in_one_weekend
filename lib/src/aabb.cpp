#include<aabb.h>

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