#include<material.h>
#include<hit.h>

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