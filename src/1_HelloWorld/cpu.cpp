#include<iostream>
#include<vector3.h>
#include<timer.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include<stb_image_write.h>

using namespace std;

int main(void) {
	const int imageWidth = 256;
	const int imageHeight = 256;

	timer t;
	t.start();

	vector<unsigned char> pngData;

	for(int i = imageHeight - 1; i >= 0; i--) {
		cout<<"\rScanlines remaining: "<<i<<' '<<flush;
		for(int j = 0; j < imageWidth; j++) {
			color pixelColor(double(j) / (imageWidth - 1), double(i) / (imageHeight - 1), 0.25);
			pixelColor.addColor(pngData);
		}
	}
	t.end();
	stbi_write_png("outputcpu.png", imageWidth, imageHeight, 3, pngData.data(), imageWidth * 3);
	cout<<"\nDone. Time Taken = "<<t<<endl;
}