#include<iostream>
#include<cmath>
#include<vector3.h>
#include<ray.h>

using namespace std;

//Find Normal Maths:
/*
normal is a unit vector perpendicular to the surface denoted by N
It is computed by the difference of the Point where Hit H and Sphere Center C
N = normalize(H - C)

Calculate distance t for Ray Hit
t = (-b +/- sqrt(Discriminant)) / (2 * a)
H = P(t)
*/

double hitSphere(point C, double r, ray P) {
	vector3 A_minus_C = P.origin() - C;
	double a = dot(P.direction(), P.direction());
	double b = 2.0 * dot(A_minus_C, P.direction());
	double c = dot(A_minus_C, A_minus_C) - r * r;
	double discriminant = b * b - 4.0 * a * c;
	if(discriminant < 0) {
		return -1.0;
	} else {
		//Use - instead of +, assuming sphere is front of camera and we want the front surface 
		return (-b - sqrt(discriminant)) / (2.0 * a);
	}
}

color rayColorFor(const ray& currentray) {
	point C = point(0.0, 0.0, -1.0);
	double t = hitSphere(C, 0.5, currentray);
	if(t > 0.0) {
		point H = currentray.at(t);
		vector3 N = H - C;
		N = N.normalize();
		N = (N + 1.0) * 0.5;
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

	double viewportHeight = 2.0;
	double viewportWidth = viewportHeight * aspectRatio;
	double focalLength = 1.0;

	point origin = point(0.0, 0.0, 0.0);
	vector3 horizontal = vector3(viewportWidth, 0.0, 0.0);
	vector3 vertical = vector3(0.0, viewportHeight, 0.0);
	vector3 lowerLeftConner = origin - horizontal/2 - vertical/2 - vector3(0, 0, focalLength);

	cout<<"P3\n"<<imageWidth<<' '<<imageHeight<<"\n255\n";
	for(int i = imageHeight - 1; i >= 0; i--) {
		cerr<<"\rScanlines remaining: "<<i<<' '<<flush;
		for(int j = 0; j < imageWidth; j++) {
			double u = (double)j / (imageWidth - 1);
			double v = (double)i / (imageHeight - 1);
			ray r(origin, lowerLeftConner + u * horizontal + v * vertical - origin);
			cout<<rayColorFor(r);
		}
	}
	cerr<<"\nDone.\n";
}