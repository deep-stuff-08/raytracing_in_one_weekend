#include<iostream>
#include<cmath>
#include<limits>
#include<vector3.h>
#include<ray.h>
#include<hit.h>
#include<camera.h>
#include<material.h>
#include<timer.h>
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
	hit_list boxes1;
	shared_ptr<material> ground = make_shared<lambertian>(color(0.48, 0.83, 0.53));

	const int boxes_per_side = 5;
	for (int i = 0; i < boxes_per_side; i++) {
		for (int j = 0; j < boxes_per_side; j++) {
			double w = 400.0;
			double x0 = -1000.0 + i * w;
			double z0 = -1000.0 + j * w;
			double y0 = 0.0;
			double x1 = x0 + w;
			double y1 = random_double(1,101);
			double z1 = z0 + w;

			boxes1.add(make_shared<cubeobj>(point(x0,y0,z0), point(x1,y1,z1), ground));
		}
	}

	hit_list objects;
	objects.add(make_shared<bvhnode>(boxes1, 0, 1));

	shared_ptr<diffuselight> light = make_shared<diffuselight>(color(7, 7, 7));
	objects.add(make_shared<quadobjxz>(123, 423, 147, 412, 554, light));

	point center1 = point(400, 400, 200);
	point center2 = center1 + vector3(30,0,0);
	shared_ptr<material> moving_sphere_material = make_shared<lambertian>(color(0.7, 0.3, 0.1));
	objects.add(make_shared<movingsphereobj>(center1, center2, moving_sphere_material, 50, 0, 1));

	objects.add(make_shared<sphereobj>(point(260, 150, 45), make_shared<dielectric>(1.5), 50));
	objects.add(make_shared<sphereobj>(point(0, 150, 145), make_shared<metal>(color(0.8, 0.8, 0.9), 1.0), 50));

	shared_ptr<sphereobj> boundary = make_shared<sphereobj>(point(360,150,145), make_shared<dielectric>(1.5), 70);
	objects.add(boundary);
	objects.add(make_shared<constantMedium>(boundary, 0.2, color(0.2, 0.4, 0.9)));
	boundary = make_shared<sphereobj>(point(0, 0, 0), make_shared<dielectric>(1.5), 5000);
	objects.add(make_shared<constantMedium>(boundary, 0.0001, color(1,1,1)));

	shared_ptr<material> emat = make_shared<lambertian>(make_shared<textureColor>("earth.jpg"));
	objects.add(make_shared<sphereobj>(point(400,200,400), emat, 100));
	shared_ptr<texture> pertext = make_shared<perlinnoiseColor>(0.1);
	objects.add(make_shared<sphereobj>(point(220,280,300), make_shared<lambertian>(pertext), 80));

	hit_list boxes2;
	shared_ptr<material> white = make_shared<lambertian>(color(.73, .73, .73));
	int ns = 1000;
	for (int j = 0; j < ns; j++) {
		boxes2.add(make_shared<sphereobj>(point::random(0,165), white, 10));
	}

	objects.add(make_shared<translate>(make_shared<rotatey>(make_shared<bvhnode>(boxes2, 0.0, 1.0), 15), vector3(-100,270,395)));

	return objects;
}

int main(void) {
	const double aspectRatio = 16.0 / 9.0;
	const int imageHeight = 360;
	const int imageWidth = static_cast<int>(imageHeight * aspectRatio);
	const int samplesPerPixel = 4000;
	const int maxDepth = 50;

	timer t;
	t.start();

	vector<unsigned char> pngData;
	
	hit_list world = generateScene();

	point lookfrom = point(478, 278, -600);
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