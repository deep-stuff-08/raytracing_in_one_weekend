#ifndef __VECTOR_3__
#define __VECTOR_3__

#include<iostream>
#include<vector>

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

	bool isNearZero() const;

	double operator[](int i) const;
	double& operator[](int i);

	vector3 operator-() const;
	vector3& operator+=(const vector3& vec);
	vector3& operator*=(const double& sca);
	vector3& operator/=(const double& sca);
	friend vector3 operator*(const vector3& vec, double sca);
	friend vector3 operator*(double sca, const vector3& vec);
	friend vector3 operator*(const vector3& vec, const vector3& vec2);
	friend vector3 operator+(const vector3& vec, const vector3& vec2);
	friend vector3 operator+(const vector3& vec, const double& sca);
	friend vector3 operator+(const double& sca, const vector3& vec);
	friend vector3 operator/(const vector3& vec, double sca);
	friend vector3 operator-(const vector3& vec, const vector3& vec2);

	double length() const;
	double length_2() const;
	vector3& normalize();

	friend double dot(vector3 v1, vector3 v2);
	friend vector3 cross(vector3 v1, vector3 v2);
	friend vector3 reflect(const vector3& v, const vector3& n);
	friend vector3 refract(const vector3& v, const vector3& n, double ratio);

	void addColor(std::vector<unsigned char>& v, int sample = 1);

	static vector3 random(double min = 0.0, double max = 1.0);
};

double random_double(double min = 0.0, double max = 1.0);
vector3 random_in_unit_disk();
vector3 random_in_unit_sphere();
vector3 random_on_unit_sphere();
vector3 random_in_unit_hemisphere(vector3 normal);

using point = vector3;
using color = vector3;

#endif