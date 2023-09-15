.PHONY: run clean

all: run

run: image.run opengl.run

image.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -lvector3 -ltimer -o image.run

clean:
	rm image.run
