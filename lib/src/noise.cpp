#include<noise.h>
#include<cmath>

noise::noise() {
	this->ranvec = new vector3[noise::numpoint];
	for(int i = 0; i < noise::numpoint; i++) {
		this->ranvec[i] = random_on_unit_sphere();
	}
	this->permx = generatePerm();
	this->permy = generatePerm();
	this->permz = generatePerm();
}

noise::~noise() {
	delete this->ranvec;
	delete this->permx;
	delete this->permy;
	delete this->permz;
}
static double perlin_interp(vector3 c[2][2][2], double u, double v, double w) {
	double uu = u*u*(3-2*u);
	double vv = v*v*(3-2*v);
	double ww = w*w*(3-2*w);

	double accum = 0.0;
	for (int i=0; i < 2; i++) {
		for (int j=0; j < 2; j++) {
			for (int k=0; k < 2; k++) {
				vector3 weight_v(u-i, v-j, w-k);
				accum += (i*uu + (1-i)*(1-uu))
						* (j*vv + (1-j)*(1-vv))
						* (k*ww + (1-k)*(1-ww))
						* dot(c[i][j][k], weight_v);
			}
		}
	}
	return accum;
}

double noise::value(const point& p) const {
	auto u = p.x() - floor(p.x());
	auto v = p.y() - floor(p.y());
	auto w = p.z() - floor(p.z());
	
	auto i = static_cast<int>(floor(p.x()));
	auto j = static_cast<int>(floor(p.y()));
	auto k = static_cast<int>(floor(p.z()));
	vector3 c[2][2][2];

	for (int di=0; di < 2; di++) {
		for (int dj=0; dj < 2; dj++) {
			for (int dk=0; dk < 2; dk++) {
				c[di][dj][dk] = this->ranvec[
					this->permx[(i+di) & 255] ^
					this->permy[(j+dj) & 255] ^
					this->permz[(k+dk) & 255]
				];
			}
		}
	}

	return perlin_interp(c, u, v, w);
}

int* noise::generatePerm() {
	int* p = new int[noise::numpoint];
	for(int i = 0; i < noise::numpoint; i++) {
		p[i] = i;
	}
	for (int i = noise::numpoint - 1; i > 0; i--) {
		int target = random_int(0, i);
		int tmp = p[i];
		p[i] = p[target];
		p[target] = tmp;
	}
	return p;
}