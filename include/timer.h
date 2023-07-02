#ifndef __TIMER__
#define __TIMER__

#include<iostream>
#include<chrono>

class timer {
private:
	std::chrono::time_point<std::chrono::steady_clock> s;
	std::chrono::time_point<std::chrono::steady_clock> e;
public:
	timer() {}
	void start();
	void end();
	friend std::ostream& operator<<(std::ostream& out, timer& t);
};

std::ostream& operator<<(std::ostream& out, std::chrono::nanoseconds time);

#endif