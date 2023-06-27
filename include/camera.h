#ifndef __CAMERA__
#define __CAMERA__

#include<vector3.h>
#include<ray.h>

class camera {
private:
	point origin;
	point lowerLeftConner;
	vector3 horizontal;
	vector3 vertical;
	double lensRadius;
	vector3 u, v, w;
	double startTime, endTime;
public:
	camera(vector3 eye, vector3 center, vector3 up, double vfov, double aspectRatio, double aperature, double focalDist, double sTime = 0.0, double eTime = 0.0);
	ray rayAt(double u, double v);
};

#endif