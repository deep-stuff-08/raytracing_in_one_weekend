#ifndef __TEXTURE__
#define __TEXTURE__

#include<vector3.h>
#include<memory>
#include<noise.h>

class texture {
public:
	virtual color value(double u, double v, const point& p) const = 0;
};

class solidColor : public texture {
private:
	color val;
public:
	solidColor() {}
	solidColor(color c) : val(c) {}
	solidColor(double r, double g, double b) : val(color(r, g, b)) {}
	virtual color value(double u, double v, const point& p) const override;
};

class checkerColor : public texture {
private:
	std::shared_ptr<texture> odd;
	std::shared_ptr<texture> even;
public:
	checkerColor() {}
	checkerColor(color o, color e) : odd(std::make_shared<solidColor>(o)), even(std::make_shared<solidColor>(e)) {}
	checkerColor(std::shared_ptr<texture> o, std::shared_ptr<texture> e) : odd(o), even(e) {}
	virtual color value(double u, double v, const point& p) const override;
};

class perlinnoiseColor : public texture {
private:
	noise noiseObj;
	double scale;
public:
	perlinnoiseColor(double frequency): scale(frequency) {}
	virtual color value(double u, double v, const point& p) const override;
};

class turbulanceColor : public texture {
private:
	noise noiseObj;
	double scale;
	int depth;
public:
	turbulanceColor(double frequency, int octaves): scale(frequency), depth(octaves) {}
	virtual color value(double u, double v, const point& p) const override;
};

class textureColor : public texture {
private:
	unsigned char* data;
	int width, height;
	int bytesPerPixel;
public:
	textureColor(std::string filename);
	~textureColor();
	virtual color value(double u, double v, const point& p) const override;
};

#endif