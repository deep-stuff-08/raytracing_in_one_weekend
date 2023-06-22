#ifndef __MATERIAL__
#define __MATERIAL

#include<ray.h>

struct hit_record;

class material {
public:
	virtual bool scatter(const ray& rayIn, const hit_record& rec, color& attenuation, ray& scattered) const = 0;
};

class lambertian : public material {
private:
	color albedo;
public:
	lambertian(const color& col) : albedo(col) {}
	virtual bool scatter(const ray& rayIn, const hit_record& rec, color& attenuation, ray& scattered) const override;
};

class metal : public material {
private:
	color albedo;
	double fuzz;
public:
	metal(const color& col, double f) : albedo(col), fuzz(f>1 ? 1 : f) {}
	virtual bool scatter(const ray& rayIn, const hit_record& rec, color& attenuation, ray& scattered) const override;
};

class dielectric : public material {
private:
	double indexOfRefraction;
public:
	dielectric(double ir) : indexOfRefraction(ir) {}
	virtual bool scatter(const ray& rayIn, const hit_record& rec, color& attenuation, ray& scattered) const override;
	static double reflectance(double costheta, double ratio);
};

#endif