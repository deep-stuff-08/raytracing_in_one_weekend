#include<hit.h>
#include<cmath>
#include<algorithm>
#include<limits>

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
	getUVs(outnormals, rec.u, rec.v);
	rec.frontFacing = dot(P.direction(), outnormals) < 0;
	rec.normal = rec.frontFacing ? outnormals : -outnormals;
	rec.normal = rec.normal.normalize();
	rec.matPtr = this->matPtr;
	return true;
}

bool sphereobj::boundingBox(double time0, double time1, aabb& outputbox) const {
	outputbox = aabb(this->center - vector3(this->radius, this->radius, this->radius), this->center + vector3(this->radius, this->radius, this->radius));
	return true;
}

void sphereobj::getUVs(const point& p, double& u, double& v) {
	double theta = acos(-p.y());
	double phi = atan2(-p.z(), p.x()) + M_PI;

	u = phi / (M_PI * 2.0);
	v = theta / M_PI;
}

bool movingsphereobj::hit(const ray& P, double t_min, double t_max, hit_record& rec) const {
	vector3 A_minus_C = P.origin() - center(P.time());
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
	vector3 outnormals = (rec.p - this->center(P.time())) / this->radius;
	rec.frontFacing = dot(P.direction(), outnormals) < 0;
	rec.normal = rec.frontFacing ? outnormals : -outnormals;
	rec.normal = rec.normal.normalize();
	rec.matPtr = this->matPtr;
	return true;
}

point movingsphereobj::center(double time) const {
	return startCenter + ((time - startTime) / (endTime - startTime)) * (endCenter - startCenter);
}

bool movingsphereobj::boundingBox(double time0, double time1, aabb& outputbox) const {
	aabb aabb0 = aabb(this->center(time0) - vector3(this->radius, this->radius, this->radius), this->center(time0) + vector3(this->radius, this->radius, this->radius));
	aabb aabb1 = aabb(this->center(time1) - vector3(this->radius, this->radius, this->radius), this->center(time1) + vector3(this->radius, this->radius, this->radius));
	outputbox = aabb0;
	return true;
}

bool quadobjxy::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
	double t = (this->k - r.origin().z()) / r.direction().z();
	if (t < t_min || t > t_max) {
		return false;
	}
	double x = r.origin().x() + t * r.direction().x();
	double y = r.origin().y() + t * r.direction().y();
	if(x < x0 || x > x1 || y < y0 || y > y1) {
		return false;
	}
	rec.u = (x - x0) / (x1 - x0);
	rec.v = (y - y0) / (y1 - y0);
	rec.t = t;
	vector3 outnormals = vector3(0, 0, 1);
	rec.frontFacing = dot(r.direction(), outnormals) < 0;
	rec.normal = rec.frontFacing ? outnormals : -outnormals;
	rec.normal = rec.normal.normalize();
	rec.matPtr = this->matPtr;
	rec.p = r.at(t);
	return true;
}

bool quadobjxy::boundingBox(double time0, double time1, aabb& outputbox) const {
	outputbox = aabb(point(this->x0, this->y0, this->k - 0.0001), point(this->x1, this->y1, this->k + 0.0001));
	return true;
}

bool quadobjyz::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
	double t = (this->k - r.origin().x()) / r.direction().x();
	if (t < t_min || t > t_max) {
		return false;
	}
	double y = r.origin().y() + t * r.direction().y();
	double z = r.origin().z() + t * r.direction().z();
	if(z < z0 || z > z1 || y < y0 || y > y1) {
		return false;
	}
	rec.u = (z - z0) / (z1 - z0);
	rec.v = (y - y0) / (y1 - y0);
	rec.t = t;
	vector3 outnormals = vector3(1, 0, 0);
	rec.frontFacing = dot(r.direction(), outnormals) < 0;
	rec.normal = rec.frontFacing ? outnormals : -outnormals;
	rec.normal = rec.normal.normalize();
	rec.matPtr = this->matPtr;
	rec.p = r.at(t);
	return true;
}

bool quadobjyz::boundingBox(double time0, double time1, aabb& outputbox) const {
	outputbox = aabb(point(this->k - 0.0001, this->y0, this->z0), point(this->k + 0.0001, this->y1, this->z1));
	return true;
}

bool quadobjxz::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
	double t = (this->k - r.origin().y()) / r.direction().y();
	if (t < t_min || t > t_max) {
		return false;
	}
	double x = r.origin().x() + t * r.direction().x();
	double z = r.origin().z() + t * r.direction().z();
	if(x < x0 || x > x1 || z < z0 || z > z1) {
		return false;
	}
	rec.u = (x - x0) / (x1 - x0);
	rec.v = (z - z0) / (z1 - z0);
	rec.t = t;
	vector3 outnormals = vector3(0, 1, 0);
	rec.frontFacing = dot(r.direction(), outnormals) < 0;
	rec.normal = rec.frontFacing ? outnormals : -outnormals;
	rec.normal = rec.normal.normalize();
	rec.matPtr = this->matPtr;
	rec.p = r.at(t);
	return true;
}

bool quadobjxz::boundingBox(double time0, double time1, aabb& outputbox) const {
	outputbox = aabb(point(this->x0, this->k - 0.0001, this->z0), point(this->x1, this->k + 0.0001, this->z1));
	return true;
}

cubeobj::cubeobj(const point& p0, const point& p1, shared_ptr<material> matPtr): boxMin(p0), boxMax(p1) {
	hit_list li;
	li.add(make_shared<quadobjxy>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), matPtr));
	li.add(make_shared<quadobjxy>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), matPtr));

	li.add(make_shared<quadobjyz>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), matPtr));
	li.add(make_shared<quadobjyz>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), matPtr));

	li.add(make_shared<quadobjxz>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), matPtr));
	li.add(make_shared<quadobjxz>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), matPtr));

	this->sides = make_shared<bvhnode>(li, 0, 0);
}

bool cubeobj::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
	return this->sides->hit(r, t_min, t_max, rec);
}

bool cubeobj::boundingBox(double time0, double time1, aabb& outputbox) const {
	outputbox = aabb(this->boxMin, this->boxMax);
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

bool hit_list::boundingBox(double time0, double time1, aabb& outputbox) const {
	if(this->objs.empty()) {
		return false;
	}

	aabb tempBox;
	bool firstBox = false;

	for(const std::shared_ptr<hitobj>& hobj : this->objs) {
		if(!hobj->boundingBox(time0, time1, tempBox)) {
			return false;
		}
		outputbox = firstBox ? tempBox : surroundingBox(outputbox, tempBox);
	}
	return true;
}

bool boxCompare(const shared_ptr<hitobj> a, const shared_ptr<hitobj> b, int axis) {
	aabb boxA, boxB;

	if (!a->boundingBox(0, 0, boxA) || !b->boundingBox(0,0, boxB)) {
		std::cerr << "No bounding box in bvh_node constructor.\n";
	}

	return boxA.min()[axis] < boxB.min()[axis];
}

bool boxCompareX(const shared_ptr<hitobj> a, const shared_ptr<hitobj> b) {
	return boxCompare(a, b, 0);
}

bool boxCompareY(const shared_ptr<hitobj> a, const shared_ptr<hitobj> b) {
	return boxCompare(a, b, 1);
}

bool boxCompareZ(const shared_ptr<hitobj> a, const shared_ptr<hitobj> b) {
	return boxCompare(a, b, 2);
}

bvhnode::bvhnode(const vector<shared_ptr<hitobj>>& srcObjects, size_t start, size_t end, double time0, double time1) {
	vector<shared_ptr<hitobj>> objects = srcObjects;

	int axis = random_int(0, 2);

	auto comparator = axis == 0 ? boxCompareX : axis == 1 ? boxCompareY : boxCompareZ;

	size_t objectSpan = end - start;

	if(objectSpan == 1) {
		this->left = this->right = objects[start];
	} else if(objectSpan == 2) {
		if(comparator(objects[start], objects[start + 1])) {
			this->left = objects[start];
			this->right = objects[start + 1];
		} else {
			this->left = objects[start + 1];	
			this->right = objects[start];
		}
	} else {
		std::sort(objects.begin() + start, objects.begin() + end, comparator);

		size_t mid = start + objectSpan / 2;
		this->left = make_shared<bvhnode>(objects, start, mid, time0, time1);
		this->right = make_shared<bvhnode>(objects, mid, end, time0, time1);
	}
	aabb boxA, boxB;

	if (!this->left->boundingBox(time0, time1, boxA) || !this->right->boundingBox(time0, time1, boxB)) {
		std::cerr << "No bounding box in bvh_node constructor.\n";
	}

	box = surroundingBox(boxA, boxB);
}

bool bvhnode::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
	if(!this->box.hit(r, t_min, t_max)) {
		return false;
	}
	bool hitLeft = this->left->hit(r, t_min, t_max, rec);
	bool hitRight = this->right->hit(r, t_min, hitLeft ? rec.t : t_max, rec);

	return hitLeft || hitRight;
}

bool bvhnode::boundingBox(double time0, double time1, aabb& outputbox) const {
	outputbox = this->box;
	return true;
}

bool translate::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
	ray movedr = ray(r.origin() - offset, r.direction(), r.time());
	if(!obj->hit(movedr, t_min, t_max, rec)) {
		return false;
	}
	rec.p += offset;
	rec.frontFacing = dot(movedr.direction(), rec.normal) < 0;
	rec.normal = rec.frontFacing ? rec.normal : -rec.normal;
	rec.normal = rec.normal.normalize();
	return true;
}

bool translate::boundingBox(double time0, double time1, aabb& outputbox) const {
	if(!obj->boundingBox(time0, time1, outputbox)) {
		return false;
	}
	outputbox = aabb(outputbox.min() + offset, outputbox.max() + offset);
	return true;
}

rotatey::rotatey(shared_ptr<hitobj> ptr, double ang) : obj(ptr) {
	double rad = ang * M_PI / 180.0f;
	this->sinTheta = sin(rad);
	this->cosTheta = cos(rad);
	this->hasBox = ptr->boundingBox(0, 0, this->box);

	point min = point(numeric_limits<double>::infinity(), numeric_limits<double>::infinity(), numeric_limits<double>::infinity());
	point max = point(-numeric_limits<double>::infinity(), -numeric_limits<double>::infinity(), -numeric_limits<double>::infinity());

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				double x = i * this->box.max().x() + (1 - i) * this->box.min().x();
				double y = j * this->box.max().y() + (1 - j) * this->box.min().y();
				double z = k * this->box.max().z() + (1 - k) * this->box.min().z();

				double newx = this->cosTheta * x + this->sinTheta * z;
				double newz = -this->sinTheta * x + this->cosTheta * z;

				vector3 tester(newx, y, newz);

				for (int c = 0; c < 3; c++) {
					min[c] = fmin(min[c], tester[c]);
					max[c] = fmax(max[c], tester[c]);
				}
			}
		}
	}
	this->box = aabb(min, max);
}

bool rotatey::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
	vector3 origin = r.origin();
	vector3 direction = r.direction();

	origin[0] = this->cosTheta * r.origin()[0] - this->sinTheta * r.origin()[2];
	origin[2] = this->sinTheta * r.origin()[0] + this->cosTheta * r.origin()[2];

	direction[0] = this->cosTheta * r.direction()[0] - this->sinTheta * r.direction()[2];
	direction[2] = this->sinTheta * r.direction()[0] + this->cosTheta * r.direction()[2];

	ray rotatedr(origin, direction, r.time());

	if (!obj->hit(rotatedr, t_min, t_max, rec)) {
		return false;
	}

	point p = rec.p;
	vector3 normal = rec.normal;

	p[0] = this->cosTheta * rec.p[0] + this->sinTheta * rec.p[2];
	p[2] = -this->sinTheta * rec.p[0] + this->cosTheta * rec.p[2];

	normal[0] = this->cosTheta * rec.normal[0] + this->sinTheta * rec.normal[2];
	normal[2] = -this->sinTheta * rec.normal[0] + this->cosTheta * rec.normal[2];

	rec.p = p;
	rec.frontFacing = dot(rotatedr.direction(), normal) < 0;
	rec.normal = rec.frontFacing ? normal : -normal;
	rec.normal = rec.normal.normalize();

	return true;
}

bool rotatey::boundingBox(double time0, double time1, aabb& outputbox) const {
	outputbox = this->box;
	return this->hasBox;
}

bool constantMedium::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
	hit_record rec1, rec2;

	if (!obj->hit(r, -numeric_limits<double>::infinity(), numeric_limits<double>::infinity(), rec1)) {
		return false;
	}

	if (!obj->hit(r, rec1.t+0.0001, numeric_limits<double>::infinity(), rec2)) {
		return false;
	}

	if (rec1.t < t_min) { rec1.t = t_min; }
	if (rec2.t > t_max) { rec2.t = t_max; }

	if (rec1.t >= rec2.t) {
		return false;
	}

	if (rec1.t < 0) {
		rec1.t = 0;
	}

	const double rayLength = r.direction().length();
	const double distanceInsideBoundary = (rec2.t - rec1.t) * rayLength;
	const double hitDistance = this->negInvDensity * log(random_double());

	if (hitDistance > distanceInsideBoundary) {
		return false;
	}

	rec.t = rec1.t + hitDistance / rayLength;
	rec.p = r.at(rec.t);

	rec.normal = vector3(1,0,0);
	rec.frontFacing = true;
	rec.matPtr = this->phaseFunction;

	return true;
}

bool constantMedium::boundingBox(double time0, double time1, aabb& outputbox) const {
	return obj->boundingBox(time0, time1, outputbox);
}
