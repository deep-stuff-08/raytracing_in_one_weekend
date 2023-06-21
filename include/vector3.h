#ifndef __VECTOR_3__
#define __VECTOR_3__

#include<iostream>

class vector3 {
protected:
	double e[3];
public:
	vector3() : e{0.0, 0.0, 0.0} {};
	vector3(double x, double y, double z) : e{x, y, z} {};

	double x() const;
	double y() const;
	double z() const;
	
	double r() const;
	double g() const;
	double b() const;

	double operator[](int i) const;
	double& operator[](int i);

	vector3 operator-() const;
	vector3& operator+=(const vector3& vec);
	vector3& operator*=(const double& sca);
	vector3& operator/=(const double& sca);
	friend vector3 operator*(const vector3& vec, double sca);
	friend vector3 operator*(double sca, const vector3& vec);
	friend vector3 operator+(const vector3& vec, const vector3& vec2);
	friend vector3 operator/(const vector3& vec, double sca);
	friend vector3 operator-(const vector3& vec, const vector3& vec2);

	double length() const;
	double length_2() const;
	vector3& normalize();

	friend double dot(vector3 v1, vector3 v2);

	friend std::ostream& operator<<(std::ostream& out, const vector3& vec);
};

using point = vector3;
using color = vector3;

#endif