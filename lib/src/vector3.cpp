#include<vector3.h>
#include<cmath>
#include<random>

using namespace std;

double vector3::x() const {
	return e[0];
}

double vector3::y() const {
	return e[1];
}

double vector3::z() const {
	return e[2];
}

double vector3::r() const {
	return e[0];
}

double vector3::g() const {
	return e[1];
}

double vector3::b() const {
	return e[2];
}

bool vector3::isNearZero() const {
	const double s = 1e-8;
	return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
}

double vector3::operator[](int i) const {
	return e[i];
}

double& vector3::operator[](int i) {
	return e[i];
}

vector3 vector3::operator-() const {
	return { -e[0], -e[1], -e[2] };
}

vector3& vector3::operator+=(const vector3& vec) {
	this->e[0] += vec.e[0];
	this->e[1] += vec.e[1];
	this->e[2] += vec.e[2];
	return *this;
}

vector3& vector3::operator*=(const double& sca) {
	this->e[0] *= sca;
	this->e[1] *= sca;
	this->e[2] *= sca;
	return *this;
}

vector3& vector3::operator/=(const double& sca) {
	(*this) *= 1.0 / sca;
	return *this;
}

vector3 operator*(const vector3& vec, double sca) {
	return vector3(vec.e[0] * sca, vec.e[1] * sca, vec.e[2] * sca);
}

vector3 operator*(double sca, const vector3& vec) {
	return vec * sca;
}

vector3 operator*(const vector3& vec, const vector3& vec2) {
	return vector3(vec.e[0] * vec2.e[0], vec.e[1] * vec2.e[1], vec.e[2] * vec2.e[2]);
}

vector3 operator+(const vector3& vec, const vector3& vec2) {
	return vector3(vec.e[0] + vec2.e[0], vec.e[1] + vec2.e[1], vec.e[2] + vec2.e[2]);
}

vector3 operator-(const vector3& vec, const vector3& vec2) {
	return vec + -vec2;
}

vector3 operator/(const vector3& vec, double sca) {
	return vec * (1/sca);
}

vector3 operator+(const vector3& vec, const double& sca) {
	return vector3(vec.e[0] + sca, vec.e[1] + sca, vec.e[2] + sca);
}

vector3 operator+(const double& sca, const vector3& vec) {
	return vec + sca;
}

double vector3::length() const {
	return sqrt(this->length_2());
}

double vector3::length_2() const {
	return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
}

vector3& vector3::normalize() {
	(*this) /= this->length();
	return (*this);
}

double dot(vector3 v1, vector3 v2) {
	return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

vector3 cross(vector3 v1, vector3 v2) {
	return vector3(
		v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1],
		v1.e[2] * v2.e[0] - v1.e[0] * v2.e[2],
		v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]
	);
}

vector3 reflect(const vector3& v, const vector3& n) {
	return v - 2.0 * dot(v, n) * n;
}

vector3 refract(const vector3& v, const vector3& n, double ratio) {
	double costheta = fmin(dot(-v, n), 1.0);
	vector3 r_perp = ratio * (v + costheta * n);
	vector3 r_parallel = -sqrt(fabs(1.0 - r_perp.length_2())) * n;
	return r_parallel + r_perp;
}

double clamp(double x, double minx, double maxx) {
	return max(min(x, 1.0), 0.0);
}

void vector3::addColor(std::vector<unsigned char>& v, int sample) {
	double r = this->e[0];
	double g = this->e[1];
	double b = this->e[2];

	if(sample > 1) {
		double scale = 1.0 / sample;
		r = sqrt(scale * r);
		g = sqrt(scale * g);
		b = sqrt(scale * b);
	}
	v.push_back(static_cast<unsigned char>(255 * clamp(r, 0, 1)));
	v.push_back(static_cast<unsigned char>(255 * clamp(g, 0, 1)));
	v.push_back(static_cast<unsigned char>(255 * clamp(b, 0, 1)));
}

double random_double(double min, double max) {
	return ((double)rand() / (double)RAND_MAX * (max - min) + min);
	// static uniform_real_distribution<double> distribution(min, max);
	// static mt19937 generator;
	// return distribution(generator);
}

int random_int(int min, int max) {
	return static_cast<int>(random_double(min, max + 1));
}

vector3 vector3::random(double min, double max) {
	return vector3(random_double(min, max), random_double(min, max), random_double(min, max));
}

vector3 random_in_unit_disk() {
	while(true) {
		vector3 v = vector3(random_double(-1.0, 1.0), random_double(-1.0, 1.0), 0.0);
		if(v.length_2() >= 1) continue;
		return v;
	}
}

vector3 random_in_unit_sphere() {
	while(true) {
		vector3 v = vector3::random(-1.0, 1.0);
		if(v.length_2() >= 1) continue;
		return v;
	}
}

vector3 random_on_unit_sphere() {
	vector3 v = vector3::random(-1.0, 1.0).normalize();
	return v;
}

vector3 random_in_unit_hemisphere(vector3 normal) {
	vector3 in_unit_sphere = random_in_unit_sphere();
	if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
		return in_unit_sphere;
	else
		return -in_unit_sphere;
}
