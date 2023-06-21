#include<iostream>
#include<vector3.h>

using namespace std;

int main(void) {
	const int imageWidth = 256;
	const int imageHeight = 256;

	cout<<"P3\n"<<imageWidth<<' '<<imageHeight<<"\n255\n";
	for(int i = imageHeight - 1; i >= 0; i--) {
		cerr<<"\rScanlines remaining: "<<i<<' '<<flush;
		for(int j = 0; j < imageWidth; j++) {
			color pixelColor(double(j) / (imageWidth - 1), double(i) / (imageHeight - 1), 0.25);
			cout<<pixelColor;
		}
	}
	cerr<<"\nDone.\n";
}