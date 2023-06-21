all: libvector3.so libray.so libhit.so

libvector3.so: src/vector3.cpp
	g++ -I ../include src/vector3.cpp -fPIC -shared -o libvector3.so

libray.so: src/ray.cpp
	g++ -I ../include src/ray.cpp -fPIC -shared -o libray.so

libhit.so: src/hit.cpp
	g++ -I ../include src/hit.cpp -fPIC -shared -o libhit.so