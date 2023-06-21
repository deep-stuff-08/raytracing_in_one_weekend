#include<iostream>
#include<cmath>
#include<limits>
#include<vector3.h>
#include<ray.h>
#include<hit.h>
#include<camera.h>

using namespace std;

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
	const int imageHeight = 360;
	const int imageWidth = static_cast<int>(imageHeight * aspectRatio);
	const int samplesPerPixel = 100;

	hit_list world;
	world.add(make_shared<sphereobj>(point(0, 0, -1), 0.5));
	world.add(make_shared<sphereobj>(point(0, -100.5, -1), 100));

	camera cam(aspectRatio, 2.0, 1.0);

	cout<<"P3\n"<<imageWidth<<' '<<imageHeight<<"\n255\n";
	for(int i = imageHeight - 1; i >= 0; i--) {
		cerr<<"\rScanlines remaining: "<<i<<' '<<flush;
		for(int j = 0; j < imageWidth; j++) {
			color pixColor;
				for(int s = 0; s < samplesPerPixel; s++) {
				double u = ((double)j + random_double(-1.0, 1.0)) / (imageWidth - 1);
				double v = ((double)i + random_double(-1.0, 1.0)) / (imageHeight - 1);
				ray r = cam.rayAt(u, v);
				pixColor += rayColorFor(r, world);
			}
			pixColor.writeColor(cout, samplesPerPixel);
		}
	}
	cerr<<"\nDone.\n";
}