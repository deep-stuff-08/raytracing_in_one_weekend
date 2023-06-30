.PHONY: clean

all: libvector3.so libray.so libhit.so libcamera.so libbasiccamera.so libmaterial.so libaabb.so libtexture.so

libvector3.so: src/vector3.cpp
	g++ -I ../include src/vector3.cpp -fPIC -shared -o libvector3.so

libray.so: src/ray.cpp
	g++ -I ../include src/ray.cpp -fPIC -shared -o libray.so

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

clean:
	rm -f *.so