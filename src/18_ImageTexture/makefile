.PHONY: run clean

all: run

run: image.run opengl.run

image.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -lvector3 -lray -ltimer -lhit -lcamera -lmaterial -laabb -ltexture -lnoise -o image.run

opengl.run:
	g++ opengl.cpp -o opengl.run

clean:
	rm opengl.run
	rm image.run
