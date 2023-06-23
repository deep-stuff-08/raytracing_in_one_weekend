#include<iostream>
#include<cmath>
#include<limits>
#include<vector3.h>
#include<ray.h>
#include<hit.h>
#include<camera.h>
#include<material.h>
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
	const int maxDepth = 20;

	vector<unsigned char> pngData;
	
	hit_list world;

	shared_ptr<material> matGround = make_shared<lambertian>(color(0.8, 0.8, 0.0));
	shared_ptr<material> matCenter = make_shared<lambertian>(color(0.1, 0.2, 0.5));
	shared_ptr<material> matLeft = make_shared<dielectric>(1.5);
	shared_ptr<material> matRight = make_shared<metal>(color(0.8, 0.6, 0.2), 0.1);

	world.add(make_shared<sphereobj>(point(0, -100.5, -1), matGround, 100));
	world.add(make_shared<sphereobj>(point(0, 0, -1), matCenter, 0.5));
	world.add(make_shared<sphereobj>(point(-1, 0, -1), matLeft, 0.5));
	world.add(make_shared<sphereobj>(point(-1, 0, -1), matLeft, -0.45));
	world.add(make_shared<sphereobj>(point(1, 0, -1), matRight, 0.5));

	vector3 lookfrom = vector3(3, 3, 2);
	vector3 lookat = vector3(0, 0, -1);
	camera cam(lookfrom, lookat, vector3(0, 1, 0), 20.0, aspectRatio, 2.0, (lookfrom - lookat).length());

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
	stbi_write_png("output.png", imageWidth, imageHeight, 3, pngData.data(), imageWidth * 3);
	cout<<"\nDone.\n";
}