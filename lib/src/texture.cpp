#include<texture.h>
#include<cmath>

color solidColor::value(double u, double v, const point &p) const {
	return val;
}

color checkerColor::value(double u, double v, const point& p) const {
	double s = sin(10.0 * p.x()) * sin(10.0 * p.y()) * sin(10.0 * p.z());
	return ((s > 0) ? this->odd : this->even)->value(u, v, p);
}
