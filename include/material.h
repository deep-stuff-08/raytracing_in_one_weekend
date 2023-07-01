#ifndef __MATERIAL__
#define __MATERIAL__

#include<ray.h>
#include<texture.h>
#include<memory>
#include<vector3.h>

struct hit_record;

class material {
public:
	virtual bool scatter(const ray& rayIn, const hit_record& rec, color& attenuation, ray& scattered) const = 0;
	virtual color emitted(double u, double v, const point& p) const;
};

class lambertian : public material {
private:
	std::shared_ptr<texture> albedo;
public:
	lambertian(const color& col) : albedo(std::make_shared<solidColor>(col)) {}
	lambertian(std::shared_ptr<texture> tex) : albedo(tex) {}
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

class diffuselight : public material {
private:
	std::shared_ptr<texture> tcolor;
public:
	diffuselight(std::shared_ptr<texture> tex): tcolor(tex) {}
	diffuselight(color col): tcolor(std::make_shared<solidColor>(col)) {}
	virtual bool scatter(const ray& rayIn, const hit_record& rec, color& attenuation, ray& scattered) const override;
	virtual color emitted(double u, double v, const point& p) const override;
};

class isotropic : public material {
private:
	std::shared_ptr<texture> albedo;
public:
	isotropic(std::shared_ptr<texture> tex): albedo(tex) {}
	isotropic(color col): albedo(std::make_shared<solidColor>(col)) {}
	virtual bool scatter(const ray& rayIn, const hit_record& rec, color& attenuation, ray& scattered) const override;
};

#endif