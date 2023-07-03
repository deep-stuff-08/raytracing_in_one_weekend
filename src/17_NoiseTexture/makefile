.PHONY: run clean

all: run

run: image.run

image.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -lvector3 -lray -laabb -ltimer -lhit -lcamera -lmaterial -laabb -ltexture -lnoise -o image.run -D TYPE=0

clean:
	rm image.run
