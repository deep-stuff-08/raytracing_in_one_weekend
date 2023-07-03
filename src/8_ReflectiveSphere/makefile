.PHONY: run clean

all: run

run: image.run

image.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -lvector3 -lray -lhit -lnoise -lbasiccamera -ltexture -lmaterial -laabb -ltimer -o image.run

clean:
	rm image.run
