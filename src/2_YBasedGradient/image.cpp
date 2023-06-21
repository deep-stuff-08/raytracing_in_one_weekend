#include<iostream>
#include<vector3.h>
#include<ray.h>

using namespace std;

color rayColorFor(const ray& currentray) {
	vector3 normalizedDirection = currentray.direction().normalize();
	double t = (normalizedDirection.y() + 1.0) * 0.5;
	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main(void) {
	const double aspectRatio = 16.0 / 9.0;
	const int imageHeight = 1920;
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