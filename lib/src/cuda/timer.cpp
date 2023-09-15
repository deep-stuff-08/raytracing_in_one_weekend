#include<timer.h>

using namespace std;

void timer::start() {
	this->s = chrono::steady_clock::now();
}

void timer::end() {
	this->e = chrono::steady_clock::now();
}

ostream& operator<<(ostream& out, chrono::nanoseconds time) {
	if(time < chrono::seconds(1)) {
		out
		<<chrono::duration_cast<chrono::milliseconds>(time).count()<<"."
		<<(chrono::duration_cast<chrono::microseconds>(time) - chrono::duration_cast<chrono::milliseconds>(time)).count()
		<<" ms";
	} else if(time < chrono::minutes(1)) {
		out
		<<chrono::duration_cast<chrono::seconds>(time).count()<<"."
		<<(chrono::duration_cast<chrono::milliseconds>(time) - chrono::duration_cast<chrono::seconds>(time)).count()
		<<" s";
	} else if(time < chrono::hours(1)) {
		out
		<<chrono::duration_cast<chrono::minutes>(time).count()<<" mins "
		<<(chrono::duration_cast<chrono::seconds>(time) - chrono::duration_cast<chrono::minutes>(time)).count()
		<<" s";
	} else {
		out
		<<chrono::duration_cast<chrono::hours>(time).count()<<" hrs "
		<<(chrono::duration_cast<chrono::minutes>(time) - chrono::duration_cast<chrono::hours>(time)).count()
		<<" mins";
	}
	return out;
}

ostream& operator<<(ostream& out, timer& t) {
	chrono::nanoseconds dur = t.e - t.s;
	out<<dur;
	return out;
}
