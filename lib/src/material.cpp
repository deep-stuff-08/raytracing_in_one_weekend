#include<material.h>
#include<hit.h>
#include<cmath>

bool lambertian::scatter(const ray& rayIn, const hit_record& rec, color& attenuation, ray& scattered) const {
	vector3 scatterDirection = rec.normal + random_on_unit_sphere();
	if(scatterDirection.isNearZero()) {
		scatterDirection = rec.normal;
	}
	scattered = ray(rec.p, scatterDirection);
	attenuation = this->albedo;
	return true;
}

bool metal::scatter(const ray& rayIn, const hit_record& rec, color& attenuation, ray& scattered) const {
	vector3 reflected = reflect(rayIn.direction().normalize(), rec.normal) + this->fuzz * random_in_unit_sphere();
	scattered = ray(rec.p, reflected);
	attenuation = albedo;
	return (dot(reflected, rec.normal) > 0);
}

bool dielectric::scatter(const ray& rayIn, const hit_record& rec, color& attenuation, ray& scattered) const {
	attenuation = color(1.0, 1.0, 1.0);
	double refractionRatio = rec.frontFacing ? (1.0 / this->indexOfRefraction) : this->indexOfRefraction;

	vector3 b = rayIn.direction().normalize();
	double costheta = fmin(dot(-b, rec.normal), 1.0);
	double sintheta = sqrt(1.0 - costheta * costheta);

	bool cannotRefract = refractionRatio * sintheta > 1.0;

	vector3 direction;
	if(cannotRefract) {
		direction = reflect(b, rec.normal);
	} else {
		direction = refract(b, rec.normal, refractionRatio);
	}
	scattered = ray(rec.p, direction);
	return true;
}

double dielectric::reflectance(double costheta, double ratio) {
	double r0 = (1 - ratio) / (1 + ratio);
	r0 = r0 * r0;
	return r0 + (1-r0) * pow((1 - costheta), 5);
}
