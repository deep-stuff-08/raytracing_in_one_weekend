#include<cudahelper.h>
#include<timer.h>
#include<vector>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include<stb_image_write.h>

using namespace std;

__global__ void render(float* fb, int maxX, int maxY) {
	int tidX = blockDim.x * blockIdx.x + threadIdx.x;
	int tidY = blockDim.y * blockIdx.y + threadIdx.y;

	if((tidX >= maxX) || (tidY >= maxY)) return;

	int pixel_index = (tidY * maxX + tidX) * 3;
	fb[pixel_index + 0] = float(tidX) / maxX;
	fb[pixel_index + 1] = float(tidY) / maxY;
	fb[pixel_index + 2] = 0.2f;
}

int main(int argc, char** argv) {
	const int imageWidth = 256;
	const int imageHeight = 256;

	// timer t;
	// t.start();

	float *fb;
	cudaerr(cudaMallocManaged(&fb, imageWidth * imageHeight * 3 * sizeof(float)));
	int tx = 8, ty = 8;
	dim3 gridSize(imageWidth / tx + 1, imageHeight / ty + 1);
	dim3 blockSize(tx, ty);
	render<<<gridSize, blockSize>>>(fb, imageWidth, imageHeight);
	cudaerr(cudaGetLastError());
	cudaerr(cudaDeviceSynchronize());

	vector<unsigned char> pngData;
	for (int j = imageHeight - 1; j >= 0; j--) {
        for (int i = 0; i < imageWidth; i++) {
            size_t pixel_index = (j * imageWidth + i) * 3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            pngData.push_back(static_cast<unsigned char>(255.99 * r));
            pngData.push_back(static_cast<unsigned char>(255.99 * g));
            pngData.push_back(static_cast<unsigned char>(255.99 * b));
        }
    }
	// t.end();
	stbi_write_png("outputcuda.png", imageWidth, imageHeight, 3, pngData.data(), imageWidth * 3);
	// cout<<"\nDone. Time Taken = "<<t<<endl;
}