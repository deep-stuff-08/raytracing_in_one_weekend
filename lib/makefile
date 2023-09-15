.PHONY: clean

<<<<<<< HEAD
CPUSRCS := $(wildcard src/cpu/*.cpp)
CPUOBJS := $(patsubst src/cpu/%.cpp, bin/cpu/%.o, $(CPUSRCS))
all: libcpuraytracer.so

libcpuraytracer.so: bin/cpu $(CPUOBJS)
	g++ $(CPUOBJS) -o libcpuraytracer.so -shared

bin/cpu: bin
	mkdir bin/cpu
=======
all: libvector3.so libray.so libhit.so libcamera.so libbasiccamera.so libmaterial.so libaabb.so libtexture.so libnoise.so libtimer.so

libvector3.so: src/vector3.cpp
	g++ -I ../include src/vector3.cpp -fPIC -shared -o libvector3.so
>>>>>>> parent of 82f39a4 (Better Library for other APIs)

libray.so: src/ray.cpp
	g++ -I ../include src/ray.cpp -fPIC -shared -o libray.so

<<<<<<< HEAD
bin/cpu/%.o : src/cpu/%.cpp
	g++ -fPIC -I ../include -c $^ -o $@

clean:
	rm -f $(CPUOBJS) libcpuraytracer.so
=======
libhit.so: src/hit.cpp
	g++ -I ../include src/hit.cpp -fPIC -shared -o libhit.so

libcamera.so: src/camera.cpp
	g++ -I ../include src/camera.cpp -fPIC -shared -o libcamera.so

libbasiccamera.so: src/basiccamera.cpp
	g++ -I ../include src/basiccamera.cpp -fPIC -shared -o libbasiccamera.so

libmaterial.so: src/material.cpp
	g++ -I ../include src/material.cpp -fPIC -shared -o libmaterial.so

libaabb.so: src/aabb.cpp
	g++ -I ../include src/aabb.cpp -fPIC -shared -o libaabb.so

libtexture.so: src/texture.cpp
	g++ -I ../include src/texture.cpp -fPIC -shared -o libtexture.so

libnoise.so: src/noise.cpp
	g++ -I ../include src/noise.cpp -fPIC -shared -o libnoise.so

libtimer.so: src/timer.cpp
	g++ -I ../include src/timer.cpp -fPIC -shared -o libtimer.so

clean:
	rm -f *.so
>>>>>>> parent of 82f39a4 (Better Library for other APIs)
