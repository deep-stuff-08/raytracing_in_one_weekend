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

#ifndef RESOURCE_PATH
#define RESOURCE_PATH ""
#endif

using namespace std;

color rayColorFor(const ray& currentray, const hitobj& world, int depth) {
	hit_record rec;
	if(depth <= 0) {
		return color();
	}
	if(!world.hit(currentray, 0.001, numeric_limits<double>::infinity(), rec)) {
		return color(0, 0, 0);
	}
	ray scattered;
	color attenuation;
	color emitted = rec.matPtr->emitted(rec.u, rec.v, rec.p);
	if(!rec.matPtr->scatter(currentray, rec, attenuation, scattered)) {
		return emitted;
	}
	return emitted + attenuation * rayColorFor(scattered, world, depth - 1);
}

hit_list generateScene() {
	hit_list world;

	shared_ptr<texture> turb = make_shared<checkerColor>(make_shared<turbulanceColor>(4.0, 9), make_shared<perlinnoiseColor>(20.0));
	shared_ptr<texture> noises = make_shared<textureColor>("earth.jpg");

	shared_ptr<material> ground = make_shared<lambertian>(turb);
	world.add(make_shared<sphereobj>(point(0, -1000, 0), ground, 1000));
	shared_ptr<material> mat1 = make_shared<lambertian>(noises);
	world.add(make_shared<sphereobj>(point(0, 1, 0), mat1, 1.0));
	
	shared_ptr<diffuselight> difflight = make_shared<diffuselight>(color(10,10,10));
	world.add(make_shared<quadobj>(3, 5, 1, 3, -2, difflight));
	
	world.add(make_shared<sphereobj>(point(0, 4.0, 0), difflight, 1.0));
	return world;
}

int main(void) {
	const double aspectRatio = 16.0 / 9.0;
	const int imageHeight = 360;
	const int imageWidth = static_cast<int>(imageHeight * aspectRatio);
	const int samplesPerPixel = 100;
	const int maxDepth = 10;

	vector<unsigned char> pngData;
	
	hit_list world = generateScene();

	vector3 lookfrom = vector3(13, 2, 3);
	vector3 lookat = vector3(0, 1, 0);
	camera cam(lookfrom, lookat, vector3(0, 1, 0), 20.0, aspectRatio, 0.1, 10.0);

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