#include<iostream>
#include<vector3.h>
#include<ray.h>
#include<timer.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include<stb_image_write.h>

using namespace std;

color rayColorFor(const ray& currentray) {
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
			color c = rayColorFor(r);
			c.addColor(pngData);
		}
	}
	t.end();
	stbi_write_png("output.png", imageWidth, imageHeight, 3, pngData.data(), imageWidth * 3);
	cout<<"\nDone. Time Taken = "<<t<<endl;
}