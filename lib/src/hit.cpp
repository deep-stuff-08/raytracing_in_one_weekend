#include<hit.h>
#include<cmath>

using namespace std;

bool sphereobj::hit(const ray& P, double t_min, double t_max, hit_record& rec) const {
	vector3 A_minus_C = P.origin() - this->center;
	double a = P.direction().length_2();
	double h = dot(A_minus_C, P.direction());
	double c = dot(A_minus_C, A_minus_C) - this->radius * this->radius;
	double discriminant = h * h - a * c;
	if(discriminant < 0) {
		return false;
	}
	double sqrtd = sqrt(discriminant);
	double root = (-h - sqrtd) / a;
	if(root < t_min || root > t_max) {
		root = (-h + sqrtd) / a;
		if(root < t_min || root > t_max) {
			return false;
		}
	}
	rec.t = root;
	rec.p = P.at(rec.t);
	vector3 outnormals = (rec.p - this->center) / this->radius;
	rec.frontFacing = dot(P.direction(), outnormals) < 0;
	rec.normal = rec.frontFacing ? outnormals : -outnormals;
	rec.normal = rec.normal.normalize();
	rec.matPtr = this->matPtr;
	return true;
}

void hit_list::add(shared_ptr<hitobj> obj) {
	this->objs.push_back(obj);
}

void hit_list::clear() {
	this->objs.clear();
}

bool hit_list::hit(const ray& P, double t_min, double t_max, hit_record& rec) const {
	hit_record tempRecord;
	bool hitAnyting = false;
	double t_closest = t_max;
	for(const shared_ptr<hitobj>& obj : this->objs) {
		if(obj->hit(P, t_min, t_closest, tempRecord)) {
			hitAnyting = true;
			t_closest = tempRecord.t;
			rec = tempRecord;
		}
	}
	return hitAnyting;
}