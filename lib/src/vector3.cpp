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

ostream& operator<<(ostream& out, const color& pixColor) {
	out<<
		static_cast<int>(255.999 * pixColor.e[0])<<' '<<
		static_cast<int>(255.999 * pixColor.e[1])<<' '<<
		static_cast<int>(255.999 * pixColor.e[2])<<'\n';
	return out;
}

double clamp(double x, double minx, double maxx) {
	return max(min(x, 1.0), 0.0);
}

void vector3::writeColor(std::ostream& out, int sample) {
	double r = this->e[0];
	double g = this->e[1];
	double b = this->e[2];

	double scale = 1.0 / sample;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);
	out<<
		static_cast<int>(256 * clamp(r, 0, 0.999))<<' '<<
		static_cast<int>(256 * clamp(g, 0, 0.999))<<' '<<
		static_cast<int>(256 * clamp(b, 0, 0.999))<<'\n';
}

double random_double(double min, double max) {
	static uniform_real_distribution<double> distribution(min, max);
	static mt19937 generator;
	return distribution(generator);
}

vector3 vector3::random(double min, double max) {
	return vector3(random_double(min, max), random_double(min, max), random_double(min, max));
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
