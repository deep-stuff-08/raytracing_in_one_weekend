#include<texture.h>
#include<cmath>

color solidColor::value(double u, double v, const point &p) const {
	return val;
}

color checkerColor::value(double u, double v, const point& p) const {
	double s = sin(10.0 * p.x()) * sin(10.0 * p.y()) * sin(10.0 * p.z());
	return ((s > 0) ? this->odd : this->even)->value(u, v, p);
}

color perlinnoiseColor::value(double u, double v, const point& p) const {
	return color(1, 1, 1) * (this->noiseObj.value(p * this->scale) * 0.5 + 0.5);
}

color turbulanceColor::value(double u, double v, const point& p) const {
	double accum = 0.0;
	auto temp_p = p;
	auto weight = 1.0;

	for (int i = 0; i < this->depth; i++) {
		accum += weight * noiseObj.value(temp_p);
		weight *= 0.5;
		temp_p *= 2;
	}

	return color(1,1,1) * fabs(accum);
}
