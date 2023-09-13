#pragma once

#include<cuda_runtime.h>
#include<iostream>

#define cudaerr(func) cudaPrintError(func, #func, __LINE__, __FILE__)

void cudaPrintError(cudaError_t error, const char* funccall, int lineno, const char* filename) {
	if(error != cudaSuccess) {
		std::cout<<"Found Cuda error "<<error<<" on function "<<funccall<<" at "<<lineno<<" : "<<filename<<std::endl;
	}
}