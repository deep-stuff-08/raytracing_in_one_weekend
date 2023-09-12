#include<iostream>
#include<cmath>
#include<limits>
#include<vector3.h>
#include<ray.h>
#include<hit.h>
#include<timer.h>
#include<basiccamera.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include<stb_image_write.h>

using namespace std;

color rayColorFor(const ray& currentray, const hitobj& world, int depth) {
	if(depth <= 0) {
		return color();
	}
	point C = point(0.0, 0.0, -1.0);
	hit_record rec;
	if(world.hit(currentray, 0.001, numeric_limits<double>::infinity(), rec)) {
#if TYPE==0
		point target = rec.p + rec.normal + random_in_unit_sphere();
#elif TYPE==1
		point target = rec.p + rec.normal + random_on_unit_sphere();
#elif TYPE==2
		point target = rec.p + rec.normal + random_in_unit_hemisphere(rec.normal);
#else
		point target = rec.p + rec.normal + random_in_unit_sphere();
#endif
		return 0.5 * rayColorFor(ray(rec.p, target - rec.p), world, depth - 1);
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

	timer t;
	t.start();

	vector<unsigned char> pngData;

	hit_list world;
	world.add(make_shared<sphereobj>(point(0, 0, -1), nullptr, 0.5));
	world.add(make_shared<sphereobj>(point(0, -100.5, -1), nullptr, 100));

	basiccamera cam(aspectRatio, 2.0, 1.0);

	for(int i = imageHeight - 1; i >= 0; i--) {
		cout<<"\rScanlines remaining: "<<i<<' '<<flush;
		for(int j = 0; j < imageWidth; j++) {
			color pixColor;
				for(int s = 0; s < samplesPerPixel; s++) {
				double u = ((double)j + random_double(-1.0, 1.0)) / (imageWidth - 1);
				double v = ((double)i + random_double(-1.0, 1.0)) / (imageHeight - 1);
				ray r = cam.rayAt(u, v);
				pixColor += rayColorFor(r, world, maxDepth);
			}
			pixColor.addColor(pngData, samplesPerPixel);
		}
	}
	t.end();
#ifdef TYPE
	string name = "output"+to_string(TYPE)+".png";
	stbi_write_png(name.c_str(), imageWidth, imageHeight, 3, pngData.data(), imageWidth * 3);
#else
	stbi_write_png("output.png", imageWidth, imageHeight, 3, pngData.data(), imageWidth * 3);
#endif
	cout<<"\nDone. Time Taken = "<<t<<endl;
}