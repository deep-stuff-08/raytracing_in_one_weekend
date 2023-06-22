.PHONY: run clean

all: run

run: image.run opengl.run

image0.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -lvector3 -lray -lhit -lbasiccamera -o image.run -D TYPE=0

image1.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -lvector3 -lray -lhit -lbasiccamera -o image.run -D TYPE=1

image2.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -lvector3 -lray -lhit -lbasiccamera -o image.run -D TYPE=2

opengl.run:
	g++ opengl.cpp -o opengl.run

clean:
	rm opengl.run
	rm image.run
