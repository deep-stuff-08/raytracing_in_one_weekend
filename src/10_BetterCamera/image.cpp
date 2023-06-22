#include<iostream>
#include<cmath>
#include<limits>
#include<vector3.h>
#include<ray.h>
#include<hit.h>
#include<camera.h>
#include<material.h>

using namespace std;

color rayColorFor(const ray& currentray, const hitobj& world, int depth) {
	if(depth <= 0) {
		return color();
	}
	point C = point(0.0, 0.0, -1.0);
	hit_record rec;
	if(world.hit(currentray, 0.001, numeric_limits<double>::infinity(), rec)) {
		ray scattered;
		color attenuation;
		if(rec.matPtr->scatter(currentray, rec, attenuation, scattered)) {
			return attenuation * rayColorFor(scattered, world, depth - 1);
		}
		return color();
	}
	vector3 normalizedDirection = currentray.direction().normalize();
	double mixt = (normalizedDirection.y() + 1.0) * 0.5;
	return (1.0 - mixt) * color(1.0, 1.0, 1.0) + mixt * color(0.5, 0.7, 1.0);
}

int main(void) {
	const double aspectRatio = 16.0 / 9.0;
	const int imageHeight = 360;
	const int imageWidth = static_cast<int>(imageHeight * aspectRatio);
	const int samplesPerPixel = 100;
	const int maxDepth = 50;

	hit_list world;

	double R = cos(M_PI_4);
	shared_ptr<material> matLeft = make_shared<lambertian>(color(0, 0, 1));
	shared_ptr<material> matRight = make_shared<lambertian>(color(1, 0, 0));

	world.add(make_shared<sphereobj>(point(-R, 0, -1), matLeft, R));
	world.add(make_shared<sphereobj>(point(R, 0, -1), matRight, R));

	camera cam(vector3(0, 0, 0), vector3(0, 0, -1), vector3(0, 1, 0), 90.0, aspectRatio, 0.0, 1.0);

	cout<<"P3\n"<<imageWidth<<' '<<imageHeight<<"\n255\n";
	for(int i = imageHeight - 1; i >= 0; i--) {
		cerr<<"\rScanlines remaining: "<<i<<' '<<flush;
		for(int j = 0; j < imageWidth; j++) {
			color pixColor;
				for(int s = 0; s < samplesPerPixel; s++) {
				double u = ((double)j + random_double(-1.0, 1.0)) / (imageWidth - 1);
				double v = ((double)i + random_double(-1.0, 1.0)) / (imageHeight - 1);
				ray r = cam.rayAt(u, v);
				pixColor += rayColorFor(r, world, maxDepth);
			}
			pixColor.writeColor(cout, samplesPerPixel);
		}
	}
	cerr<<"\nDone.\n";
}