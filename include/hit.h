#ifndef __HIT__
#define __HIT__

#include<vector>
#include<memory>
#include<ray.h>
#include<aabb.h>

class material;

struct hit_record {
	point p;
	vector3 normal;
	std::shared_ptr<material> matPtr;
	double t;
	bool frontFacing;
	double u;
	double v;
};

class hitobj {
public:
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
	virtual bool boundingBox(double time0, double time1, aabb& outputbox) const = 0;
};

class sphereobj : public hitobj {
private:
	point center;
	double radius;
	std::shared_ptr<material> matPtr;
	static void getUVs(const point& p, double& u, double& v);
public:
	sphereobj() : radius(0) {}
	sphereobj(point p, std::shared_ptr<material> mat, double r) : center(p), radius(r), matPtr(mat) {}
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
	virtual bool boundingBox(double time0, double time1, aabb& outputbox) const override;
};

class movingsphereobj : public hitobj {
private:
	point startCenter, endCenter;
	double radius;
	std::shared_ptr<material> matPtr;
	double startTime, endTime;
public:
	movingsphereobj() : radius(0) {}
	movingsphereobj(point p0, point p1, std::shared_ptr<material> mat, double r, double t0, double t1) : startCenter(p0), endCenter(p1), radius(r), matPtr(mat), startTime(t0), endTime(t1) {}
	point center(double time) const;
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
	virtual bool boundingBox(double time0, double time1, aabb& outputbox) const override;
};

class quadobjxy : public hitobj {
private:
	std::shared_ptr<material> matPtr;
	double x0, x1, y0, y1, k;
public:
	quadobjxy() {}
	quadobjxy(double _x0, double _x1, double _y0, double _y1, double _k, std::shared_ptr<material> mat): x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), matPtr(mat) {}
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
	virtual bool boundingBox(double time0, double time1, aabb& outputbox) const override;
};

class quadobjyz : public hitobj {
private:
	std::shared_ptr<material> matPtr;
	double y0, y1, z0, z1, k;
public:
	quadobjyz() {}
	quadobjyz(double _y0, double _y1, double _z0, double _z1, double _k, std::shared_ptr<material> mat): y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), matPtr(mat) {}
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
	virtual bool boundingBox(double time0, double time1, aabb& outputbox) const override;
};

class quadobjxz : public hitobj {
private:
	std::shared_ptr<material> matPtr;
	double x0, x1, z0, z1, k;
public:
	quadobjxz() {}
	quadobjxz(double _x0, double _x1, double _z0, double _z1, double _k, std::shared_ptr<material> mat): x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), matPtr(mat) {}
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
	virtual bool boundingBox(double time0, double time1, aabb& outputbox) const override;
};

class bvhnode;

class hit_list : public hitobj {
private:
	std::vector<std::shared_ptr<hitobj>> objs;
public:
	hit_list() {}
	void clear();
	void add(std::shared_ptr<hitobj> obj);
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
	virtual bool boundingBox(double time0, double time1, aabb& outputbox) const override;
	friend bvhnode;
};

class bvhnode : public hitobj {
private:
	std::shared_ptr<hitobj> left;
	std::shared_ptr<hitobj> right;
	aabb box;
public:
	bvhnode() {}
	bvhnode(const hit_list& list, double time0, double time1) : bvhnode(list.objs, 0, list.objs.size(), time0, time1) {}
	bvhnode(const std::vector<std::shared_ptr<hitobj>>& srcObjects, size_t start, size_t end, double time0, double time1);
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
	virtual bool boundingBox(double time0, double time1, aabb& outputbox) const override;
};

#endif