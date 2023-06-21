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

	double length() const;
	double length_2() const;
};

class point : public vector3 {

};

class color : public vector3 {
public:
	color() : vector3() {}
	color(double r, double g, double b) : vector3(r, g, b) {}
	friend std::ostream& operator<<(std::ostream& out, color& vec);
};

#endif