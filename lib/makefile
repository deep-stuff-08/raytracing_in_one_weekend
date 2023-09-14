.PHONY: clean

DIRGUARD=@mkdir -p $(@D)

CPUSRCS := $(wildcard src/cpu/*.cpp)
CPUOBJS := $(patsubst src/cpu/%.cpp, bin/cpu/%.o, $(CPUSRCS))

CUDASRCS := $(wildcard src/cuda/*.cu)
CUDAOBJS := $(patsubst src/cuda/%.cu, bin/cuda/%.o, $(CUDASRCS))

all: libcpuraytracer.so libcudaraytracer.a

libcpuraytracer.so: $(CPUOBJS)
	g++ $(CPUOBJS) -o libcpuraytracer.so -shared

libcudaraytracer.a: $(CUDAOBJS)
	nvcc $(CUDAOBJS) -lib -o libcudaraytracer.a

bin/cpu/%.o : src/cpu/%.cpp
	$(DIRGUARD)
	g++ -fPIC -I ../include -c $^ -o $@

bin/cuda/%.o : src/cuda/%.cu
	$(DIRGUARD)
	nvcc -rdc=true -Xcompiler=-fPIC -I ../include -c $^ -o $@

clean:
	rm -f $(CPUOBJS) $(CUDAOBJS) libcpuraytracer.so libcudaraytracer.so