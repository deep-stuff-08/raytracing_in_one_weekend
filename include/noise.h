#ifndef __NOISE__
#define __NOISE__

#include<vector3.h>

class noise {
private:
	static const int numpoint = 256;
	vector3* ranvec;
	int* permx;
	int* permy;
	int* permz;
public:
	noise();
	~noise();
	double value(const point& p) const;
	static int* generatePerm();
};

#endif