#include<iostream>
#include<cmath>
#include<limits>
#include<vector3.h>
#include<ray.h>
#include<hit.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include<stb_image_write.h>

using namespace std;

//Optimize Maths:
/*
substitute for performance
b.b = |b|^2
(A-C).(A-C) = |(A-C)|^2

let h = b/2 or b = 2h
so b = 2.0 * dot(A-C, b) => h = dot(A-C, b)

also replace in quadractic eqn
(-b +/- sqrt(b * b - 4 * a * c)) / (2.0 * a)
(-2h +/- sqrt(2h * 2h - 4 * a * c)) / (2.0 * a)
(-2h +/- 2(h * h - ac)) / (2.0  * a)
(-h +/- (h * h - ac)) / a
*/

// double hitSphere(point C, double r, ray P) {
// 	vector3 A_minus_C = P.origin() - C;
// 	double a = P.direction().length_2();
// 	double h = dot(A_minus_C, P.direction());
// 	double c = dot(A_minus_C, A_minus_C) - r * r;
// 	double discriminant = h * h - a * c;
// 	if(discriminant < 0) {
// 		return -1.0;
// 	} else {
// 		//Used - instead of +, assuming sphere is front of camera and we want the front surface 
// 		return (-h - sqrt(discriminant)) / a;
// 	}
// }



color rayColorFor(const ray& currentray, const hitobj& world) {
	point C = point(0.0, 0.0, -1.0);
	hit_record rec;
	if(world.hit(currentray, 0, numeric_limits<double>::infinity(), rec)) {
		vector3 N = (rec.normal + 1.0) * 0.5;
		return N;
	}
	vector3 normalizedDirection = currentray.direction().normalize();
	double mixt = (normalizedDirection.y() + 1.0) * 0.5;
	return (1.0 - mixt) * color(1.0, 1.0, 1.0) + mixt * color(0.5, 0.7, 1.0);
}

int main(void) {
	const double aspectRatio = 16.0 / 9.0;
	const int imageHeight = 1080;
	const int imageWidth = static_cast<int>(imageHeight * aspectRatio);

	vector<unsigned char> pngData;
	
	hit_list world;
	world.add(make_shared<sphereobj>(point(0, 0, -1), nullptr, 0.5));
	world.add(make_shared<sphereobj>(point(0, -100.5, -1), nullptr, 100));

	double viewportHeight = 2.0;
	double viewportWidth = viewportHeight * aspectRatio;
	double focalLength = 1.0;

	point origin = point(0.0, 0.0, 0.0);
	vector3 horizontal = vector3(viewportWidth, 0.0, 0.0);
	vector3 vertical = vector3(0.0, viewportHeight, 0.0);
	vector3 lowerLeftConner = origin - horizontal/2 - vertical/2 - vector3(0, 0, focalLength);

	for(int i = imageHeight - 1; i >= 0; i--) {
		cout<<"\rScanlines remaining: "<<i<<' '<<flush;
		for(int j = 0; j < imageWidth; j++) {
			double u = (double)j / (imageWidth - 1);
			double v = (double)i / (imageHeight - 1);
			ray r(origin, lowerLeftConner + u * horizontal + v * vertical - origin);
			rayColorFor(r, world).addColor(pngData);
		}
	}
	stbi_write_png("output.png", imageWidth, imageHeight, 3, pngData.data(), imageWidth * 3);
	cout<<"\nDone.\n";
}