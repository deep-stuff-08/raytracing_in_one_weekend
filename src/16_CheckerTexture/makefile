.PHONY: run clean

all: run

run: image.run

image.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -lvector3 -lray -ltexture -ltimer -lnoise -lhit -lcamera -lmaterial -laabb -ltexture -o image.run

clean:
	rm image.run
