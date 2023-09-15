.PHONY: run clean

all: run

run: image.run

image.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -lvector3 -laabb -ltexture -lnoise -ltimer -lray -lhit -lcamera -lmaterial -o image.run -D TYPE=0

clean:
	rm image.run
