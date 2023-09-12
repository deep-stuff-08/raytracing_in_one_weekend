#ifndef __BASIC_CAMERA__
#define __BASIC_CAMERA__

#include<vector3.h>
#include<ray.h>

class basiccamera {
private:
	point origin;
	point lowerLeftConner;
	vector3 horizontal;
	vector3 vertical;
public:
	basiccamera(double aspectRatio, double viewport, double focalLenght);
	ray rayAt(double u, double v);
};

#endif