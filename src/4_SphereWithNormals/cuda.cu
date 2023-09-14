#include<cudahelper.h>
#include<timer.h>
#include<vector3.cuh>
#include<ray.cuh>
#include<vector>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include<stb_image_write.h>

using namespace std;

__device__ float hitSphere(point C, float r, ray P) {
	vector3 A_minus_C = P.origin() - C;
	float a = dot(P.direction(), P.direction());
	float b = 2.0 * dot(A_minus_C, P.direction());
	float c = dot(A_minus_C, A_minus_C) - r * r;
	float discriminant = b * b - 4.0 * a * c;
	if(discriminant < 0) {
		return -1.0;
	} else {
		//Use - instead of +, assuming sphere is front of camera and we want the front surface 
		return (-b - sqrt(discriminant)) / (2.0 * a);
	}
}

__device__ color colorForRay(ray const& r) {
	point C = point(0.0, 0.0, -1.0);
	float t = hitSphere(C, 0.5, r);
	if(t > 0.0) {
		point H = r.at(t);
		vector3 N = H - C;
		N = N.normalize();
		N = (N + 1.0) * 0.5;
		return N;
	}
	vector3 normalizedDirection = r.direction().normalize();
	float mixt = (normalizedDirection.y() + 1.0) * 0.5;
	return (1.0 - mixt) * color(1.0, 1.0, 1.0) + mixt * color(0.5, 0.7, 1.0);
}

__global__ void render(color* fb, int maxX, int maxY, vector3 lowerLeftCorner, vector3 horizontal, vector3 vertical, point origin) {
	int tidX = blockDim.x * blockIdx.x + threadIdx.x;
	int tidY = blockDim.y * blockIdx.y + threadIdx.y;

	if((tidX >= maxX) || (tidY >= maxY)) return;

	int pixel_index = tidY * maxX + tidX;
	float u = float(tidX) / maxX;
	float v = float(tidY) / maxY;
	ray r = ray(origin, lowerLeftCorner + u * horizontal + v * vertical);
	fb[pixel_index] = colorForRay(r);
}

int main(int argc, char** argv) {
	const float aspectRatio = 16.0 / 9.0;
	const int imageHeight = 1080;
	const int imageWidth = static_cast<int>(imageHeight * aspectRatio);

	float viewportHeight = 2.0;
	float viewportWidth = viewportHeight * aspectRatio;
	float focalLength = 1.0;

	point origin = point(0.0, 0.0, 0.0);
	vector3 horizontal = vector3(viewportWidth, 0.0, 0.0);
	vector3 vertical = vector3(0.0, viewportHeight, 0.0);
	vector3 lowerLeftConner = origin - horizontal/2 - vertical/2 - vector3(0, 0, focalLength);

	timer t;
	t.start();

	color *fb;
	cudaerr(cudaMallocManaged(&fb, imageWidth * imageHeight * 3 * sizeof(float)));
	int tx = 8, ty = 8;
	dim3 gridSize(imageWidth / tx + 1, imageHeight / ty + 1);
	dim3 blockSize(tx, ty);
	render<<<gridSize, blockSize>>>(fb, imageWidth, imageHeight, lowerLeftConner, horizontal, vertical, origin);
	cudaerr(cudaGetLastError());
	cudaerr(cudaDeviceSynchronize());

	vector<unsigned char> pngData;
	for (int j = imageHeight - 1; j >= 0; j--) {
		for (int i = 0; i < imageWidth; i++) {
			size_t pixel_index = j * imageWidth + i;
			fb[pixel_index].addColor(pngData);
		}
	}
	t.end();
	stbi_write_png("outputcuda.png", imageWidth, imageHeight, 3, pngData.data(), imageWidth * 3);
	cout<<"Done. Time Taken = "<<t<<endl;

	cudaFree(fb);
}