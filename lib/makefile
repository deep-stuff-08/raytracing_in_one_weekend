.PHONY: vector3

all: vector3

vector3: libvector3.so

libvector3.so: src/vector3.cpp
	g++ -I ../include src/vector3.cpp -fPIC -shared -o libvector3.so