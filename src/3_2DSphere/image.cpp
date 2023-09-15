#include<iostream>
#include<vector3.h>
#include<timer.h>
#include<ray.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include<stb_image_write.h>

using namespace std;

//Sphere Maths:
/*
|P|^2 = r^2 then Point P lies on the surface of Sphere with r radius
|P|^2 = P.P = r^2

modify formula to account for a sphere with different center C
(P-C).(P-C) = r^2


substitute P to the ray function for the distance t
(P(t) - C).(P(t) - C) = r^2

Expand the ray form
(A + tb - C).(A + tb - C) = r^2

Simplify using vector algebra
=> (A.A) + (A.tb) - (A.C) + (A.tb) + (tb.tb) - (tb.C) - (A.C) - (tb.C) + (C.C) = r^2.......Move terms around
=> (tb.tb) + (A.tb) + (A.tb) - (tb.C) - (tb.C) - (A.C) - (A.C) + (A.A) + (C.C) - r^2 = 0
=> t^2b.b + 2(A.tb) - 2(tb.C) - 2(A.C) + |A|^2 + |C|^2 - r^2 = 0.................use product rule (A-B).(A-B) = |A|^2 + |B|^2 + 2(A.B)
=> b.bt^2 + 2(A-C).tb + (A-C).(A-C) - r^2 --final formula 

Based on this quadratic eqn a, b, c = as follows
a = (b.b)
b = 2.0 * ((A-C).b)
c = (A-C).(A-C) - r^2
*/

bool didRayHitSphere(point C, double r, ray P) {
	vector3 A_minus_C = P.origin() - C;
	double a = dot(P.direction(), P.direction());
	double b = 2.0 * dot(A_minus_C, P.direction());
	double c = dot(A_minus_C, A_minus_C) - r * r;

	//Discriminant of Quadratic Eqn Formula - b^2 + 4ac
	double discriminant = b * b - 4.0 * a * c;
	return discriminant > 0;
}

color rayColorFor(const ray& currentray) {
	if(didRayHitSphere(point(0.0, 0.0, -1.0), 0.5, currentray)) {
		return color(1.0, 0.0, 0.0);
	}
	vector3 normalizedDirection = currentray.direction().normalize();
	double t = (normalizedDirection.y() + 1.0) * 0.5;
	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main(void) {
	const double aspectRatio = 16.0 / 9.0;
	const int imageHeight = 1080;
	const int imageWidth = static_cast<int>(imageHeight * aspectRatio);

	timer t;
	t.start();

	vector<unsigned char> pngData;

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
			rayColorFor(r).addColor(pngData);
		}
	}
	t.end();
	stbi_write_png("output.png", imageWidth, imageHeight, 3, pngData.data(), imageWidth * 3);
	cout<<"\nDone. Time Taken = "<<t<<endl;
}