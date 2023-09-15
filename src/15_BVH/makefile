.PHONY: run clean

all: run

run: image.run

image.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -lvector3 -lray -lhit -lcamera -lmaterial -laabb -ltimer -ltexture -lnoise -o image.run

clean:
	rm image.run
