#include<texture.h>
#include<cmath>
#define STB_IMAGE_IMPLEMENTATION
#include<stb_image.h>

using namespace std;

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

	return color(1,1,1) * 0.5 * (1 + sin(this->scale * p.z() + 10 * fabs(accum)));
}

textureColor::textureColor(string filename) {
	this->data = stbi_load(filename.c_str(), &this->width, &this->height, &this->bytesPerPixel, 3);

	if(!this->data) {
		cerr<<"'"<<filename<<"' image could not be loaded."<<endl;
	}
}

textureColor::~textureColor() {
	stbi_image_free(this->data);
}

color textureColor::value(double u, double v, const point& p) const {
	if(!this->data) {
		return color(0, 0, 0);
	}
	u = clamp(u, 0, 1);
	v = 1.0 - clamp(v, 0, 1);

	int i = static_cast<int>(u * this->width - 1);
	int j = static_cast<int>(v * this->height - 1);

	const auto color_scale = 1.0 / 255.0;
	unsigned char* pixel = data + (j * this->bytesPerPixel * this->width) + (i * this->bytesPerPixel);
	return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
}