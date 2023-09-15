.PHONY: clean

DIRGUARD=@mkdir -p $(@D)

CPUSRCS := $(wildcard src/cpu/*.cpp)
CPUOBJS := $(patsubst src/cpu/%.cpp, bin/cpu/%.o, $(CPUSRCS))

CUDASRCS := $(wildcard src/cuda/*.cpp)
CUDAOBJS := $(patsubst src/cuda/%.cpp, bin/cuda/%.o, $(CUDASRCS))

all: libcpuraytracer.so libcudaraytracer.so

libcpuraytracer.so: $(CPUOBJS)
	g++ $(CPUOBJS) -o libcpuraytracer.so -shared

libcudaraytracer.so: $(CUDAOBJS)
	nvcc $(CUDAOBJS) -o libcudaraytracer.so -shared

bin/cpu/%.o : src/cpu/%.cpp
	$(DIRGUARD)
	g++ -fPIC -I ../include -c $^ -o $@

bin/cuda/%.o : src/cuda/%.cpp
	$(DIRGUARD)
	nvcc -I ../include -c $^ -o $@

clean:
	rm -f $(CPUOBJS) $(CUDAOBJS) libcpuraytracer.so libcudaraytracer.so