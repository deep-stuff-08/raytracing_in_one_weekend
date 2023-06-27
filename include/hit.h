#ifndef __HIT__
#define __HIT__

#include<vector>
#include<memory>
#include<ray.h>

class material;

struct hit_record {
	point p;
	vector3 normal;
	std::shared_ptr<material> matPtr;
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
	std::shared_ptr<material> matPtr;
public:
	sphereobj() : radius(0) {}
	sphereobj(point p, std::shared_ptr<material> mat, double r) : center(p), radius(r), matPtr(mat) {}
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
};

class movingsphereobj : public hitobj {
	point startCenter, endCenter;
	double radius;
	std::shared_ptr<material> matPtr;
	double startTime, endTime;
public:
	movingsphereobj() : radius(0) {}
	movingsphereobj(point p0, point p1, std::shared_ptr<material> mat, double r, double t0, double t1) : startCenter(p0), endCenter(p1), radius(r), matPtr(mat), startTime(t0), endTime(t1) {}
	point center(double time) const;
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