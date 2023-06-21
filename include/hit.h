#ifndef __HIT__
#define __HIT__

#include<vector>
#include<memory>
#include<ray.h>

struct hit_record {
	point p;
	vector3 normal;
	double t;
	bool frontFacing;
};

class hitobj {
public:
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};

class sphereobj : public hitobj {
private:
	point center;
	double radius;
public:
	sphereobj() : radius(0) {}
	sphereobj(point p, double r) : center(p), radius(r) {}
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
};

class hit_list : public hitobj {
private:
	std::vector<std::shared_ptr<hitobj>> objs;
public:
	hit_list() {}
	void clear();
	void add(std::shared_ptr<hitobj> obj);
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
};

#endif