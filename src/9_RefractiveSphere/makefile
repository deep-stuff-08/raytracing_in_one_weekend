.PHONY: run clean

all: run

run: image.run

image.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -ltexture -laabb -ltimer -lnoise -lvector3 -lray -lhit -lbasiccamera -lmaterial -o image.run

clean:
	rm image.run
