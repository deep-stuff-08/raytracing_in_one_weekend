#ifndef __VECTOR_3__
#define __VECTOR_3__

#include<iostream>
#include<vector>

class vector3 {
protected:
	float e[3];
public:
	__host__ __device__ vector3();
	__host__ __device__ vector3(float x, float y, float z);

	__host__ __device__ float x() const;
	__host__ __device__ float y() const;
	__host__ __device__ float z() const;
	
	__host__ __device__ float r() const;
	__host__ __device__ float g() const;
	__host__ __device__ float b() const;

	__host__ __device__ bool isNearZero() const;

	__host__ __device__ float operator[](int i) const;
	__host__ __device__ float& operator[](int i);

	__host__ __device__ vector3 operator-() const;
	__host__ __device__ vector3& operator+=(const vector3& vec);
	__host__ __device__ vector3& operator*=(const float& sca);
	__host__ __device__ vector3& operator/=(const float& sca);
	__host__ __device__ friend vector3 operator*(const vector3& vec, float sca);
	__host__ __device__ friend vector3 operator*(float sca, const vector3& vec);
	__host__ __device__ friend vector3 operator*(const vector3& vec, const vector3& vec2);
	__host__ __device__ friend vector3 operator+(const vector3& vec, const vector3& vec2);
	__host__ __device__ friend vector3 operator+(const vector3& vec, const float& sca);
	__host__ __device__ friend vector3 operator+(const float& sca, const vector3& vec);
	__host__ __device__ friend vector3 operator/(const vector3& vec, float sca);
	__host__ __device__ friend vector3 operator-(const vector3& vec, const vector3& vec2);

	__host__ __device__ float length() const;
	__host__ __device__ float length_2() const;
	__host__ __device__ vector3& normalize();

	__host__ __device__ friend float dot(vector3 v1, vector3 v2);
	// __host__ __device__ friend vector3 cross(vector3 v1, vector3 v2);
	// __host__ __device__ friend vector3 reflect(const vector3& v, const vector3& n);
	// __host__ __device__ friend vector3 refract(const vector3& v, const vector3& n, float ratio);

	__host__ void addColor(std::vector<unsigned char>& v, int sample = 1);

	// __host__ __device__ static vector3 random(float min = 0.0, float max = 1.0);
};

using point = vector3;
using color = vector3;

vector3::vector3() : e{0.0, 0.0, 0.0} {

}

vector3::vector3(float x, float y, float z) : e{x, y, z} {

}

float vector3::x() const {
	return e[0];
}

float vector3::y() const {
	return e[1];
}

float vector3::z() const {
	return e[2];
}

float vector3::r() const {
	return e[0];
}

float vector3::g() const {
	return e[1];
}

float vector3::b() const {
	return e[2];
}

// bool vector3::isNearZero() const {
// 	const float s = 1e-8;
// 	return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
// }

float vector3::operator[](int i) const {
	return e[i];
}

float& vector3::operator[](int i) {
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

vector3& vector3::operator*=(const float& sca) {
	this->e[0] *= sca;
	this->e[1] *= sca;
	this->e[2] *= sca;
	return *this;
}

vector3& vector3::operator/=(const float& sca) {
	(*this) *= 1.0 / sca;
	return *this;
}

vector3 operator*(const vector3& vec, float sca) {
	return vector3(vec.e[0] * sca, vec.e[1] * sca, vec.e[2] * sca);
}

vector3 operator*(float sca, const vector3& vec) {
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

vector3 operator/(const vector3& vec, float sca) {
	return vec * (1/sca);
}

vector3 operator+(const vector3& vec, const float& sca) {
	return vector3(vec.e[0] + sca, vec.e[1] + sca, vec.e[2] + sca);
}

vector3 operator+(const float& sca, const vector3& vec) {
	return vec + sca;
}

float vector3::length() const {
	return sqrt(this->length_2());
}

float vector3::length_2() const {
	return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
}

vector3& vector3::normalize() {
	(*this) /= this->length();
	return (*this);
}

float dot(vector3 v1, vector3 v2) {
	return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

// vector3 cross(vector3 v1, vector3 v2) {
// 	return vector3(
// 		v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1],
// 		v1.e[2] * v2.e[0] - v1.e[0] * v2.e[2],
// 		v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]
// 	);
// }

// vector3 reflect(const vector3& v, const vector3& n) {
// 	return v - 2.0 * dot(v, n) * n;
// }

// vector3 refract(const vector3& v, const vector3& n, float ratio) {
// 	float costheta = fmin(dot(-v, n), 1.0);
// 	vector3 r_perp = ratio * (v + costheta * n);
// 	vector3 r_parallel = -sqrt(fabs(1.0 - r_perp.length_2())) * n;
// 	return r_parallel + r_perp;
// }

float clamp(float x, float minx, float maxx) {
	return max(min(x, 1.0), 0.0);
}

void vector3::addColor(std::vector<unsigned char>& v, int sample) {
	float r = this->e[0];
	float g = this->e[1];
	float b = this->e[2];

	if(sample > 1) {
		float scale = 1.0 / sample;
		r = sqrt(scale * r);
		g = sqrt(scale * g);
		b = sqrt(scale * b);
	}
	v.push_back(static_cast<unsigned char>(255 * clamp(r, 0, 1)));
	v.push_back(static_cast<unsigned char>(255 * clamp(g, 0, 1)));
	v.push_back(static_cast<unsigned char>(255 * clamp(b, 0, 1)));
}

// float random_float(float min, float max) {
// 	return ((float)rand() / (float)RAND_MAX * (max - min) + min);
// 	// static uniform_real_distribution<float> distribution(min, max);
// 	// static mt19937 generator;
// 	// return distribution(generator);
// }

// int random_int(int min, int max) {
// 	return static_cast<int>(random_float(min, max + 1));
// }

// vector3 vector3::random(float min, float max) {
// 	return vector3(random_float(min, max), random_float(min, max), random_float(min, max));
// }

// vector3 random_in_unit_disk() {
// 	while(true) {
// 		vector3 v = vector3(random_float(-1.0, 1.0), random_float(-1.0, 1.0), 0.0);
// 		if(v.length_2() >= 1) continue;
// 		return v;
// 	}
// }

// vector3 random_in_unit_sphere() {
// 	while(true) {
// 		vector3 v = vector3::random(-1.0, 1.0);
// 		if(v.length_2() >= 1) continue;
// 		return v;
// 	}
// }

// vector3 random_on_unit_sphere() {
// 	vector3 v = vector3::random(-1.0, 1.0).normalize();
// 	return v;
// }

// vector3 random_in_unit_hemisphere(vector3 normal) {
// 	vector3 in_unit_sphere = random_in_unit_sphere();
// 	if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
// 		return in_unit_sphere;
// 	else
// 		return -in_unit_sphere;
// }

#endif