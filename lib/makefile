.PHONY: clean

CPUSRCS := $(wildcard src/cpu/*.cpp)
CPUOBJS := $(patsubst src/cpu/%.cpp, bin/cpu/%.o, $(CPUSRCS))
all: libcpuraytracer.so

libcpuraytracer.so: bin/cpu $(CPUOBJS)
	g++ $(CPUOBJS) -o libcpuraytracer.so -shared

bin/cpu: bin
	mkdir bin/cpu

bin:
	mkdir bin

bin/cpu/%.o : src/cpu/%.cpp
	g++ -fPIC -I ../include -c $^ -o $@

clean:
	rm -f $(CPUOBJS) libcpuraytracer.so