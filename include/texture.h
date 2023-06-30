#ifndef __TEXTURE__
#define __TEXTURE__

#include<vector3.h>
#include<memory>

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

#endif