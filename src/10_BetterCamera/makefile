.PHONY: run clean

all: run

run: image.run

image.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -lvector3 -ltexture -laabb -lnoise -ltimer -lray -lhit -lcamera -lmaterial -o image.run

clean:
	rm image.run
