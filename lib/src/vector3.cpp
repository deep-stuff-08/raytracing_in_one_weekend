#include<vector3.h>

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

double vector3::length() const {
	return this->length_2();
}

double vector3::length_2() const {
	return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
}

ostream& operator<<(ostream& out, color& pixColor) {
	out<<
		static_cast<int>(255.999 * pixColor.e[0])<<' '<<
		static_cast<int>(255.999 * pixColor.e[1])<<' '<<
		static_cast<int>(255.999 * pixColor.e[2])<<'\n';
	return out;
}
