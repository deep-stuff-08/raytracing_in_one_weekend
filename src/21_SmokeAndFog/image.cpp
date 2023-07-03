#include<iostream>
#include<cmath>
#include<limits>
#include<vector3.h>
#include<ray.h>
#include<hit.h>
#include<timer.h>
#include<camera.h>
#include<material.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include<stb_image_write.h>

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

	auto red = make_shared<lambertian>(color(.65, .05, .05));
	auto white = make_shared<lambertian>(color(.73, .73, .73));
	auto green = make_shared<lambertian>(color(.12, .45, .15));
	auto light = make_shared<diffuselight>(color(7, 7, 7));

	world.add(make_shared<quadobjyz>(0, 555, 0, 555, 555, green));
	world.add(make_shared<quadobjyz>(0, 555, 0, 555, 0, red));
	world.add(make_shared<quadobjxz>(113, 443, 127, 432, 554, light));
	world.add(make_shared<quadobjxz>(0, 555, 0, 555, 555, white));
	world.add(make_shared<quadobjxz>(0, 555, 0, 555, 0, white));
	world.add(make_shared<quadobjxy>(0, 555, 0, 555, 555, white));

	shared_ptr<hitobj> box1 = make_shared<cubeobj>(point(0,0,0), point(165,330,165), white);
	box1 = make_shared<rotatey>(box1, 15);
	box1 = make_shared<translate>(box1, vector3(265,0,295));

	shared_ptr<hitobj> box2 = make_shared<cubeobj>(point(0,0,0), point(165,165,165), white);
	box2 = make_shared<rotatey>(box2, -18);
	box2 = make_shared<translate>(box2, vector3(130,0,65));

	world.add(make_shared<constantMedium>(box1, 0.01, color(0,0,0)));
	world.add(make_shared<constantMedium>(box2, 0.01, color(1,1,1)));

	return world;
}

int main(void) {
	const double aspectRatio = 16.0 / 9.0;
	const int imageHeight = 360;
	const int imageWidth = static_cast<int>(imageHeight * aspectRatio);
	const int samplesPerPixel = 100;
	const int maxDepth = 10;

	timer t;
	t.start();

	vector<unsigned char> pngData;
	
	hit_list world = generateScene();

	point lookfrom = point(278, 278, -800);
	point lookat = point(278, 278, 0);
	camera cam(lookfrom, lookat, vector3(0, 1, 0), 40.0, aspectRatio, 0.1, 10.0);

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
	stbi_write_png("output.png", imageWidth, imageHeight, 3, pngData.data(), imageWidth * 3);
	cout<<"\nDone. Time Taken = "<<t<<endl;
}