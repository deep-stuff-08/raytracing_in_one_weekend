.PHONY: run clean

all: run

run: image.run

image.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -lvector3 -lray -lhit -lbasiccamera -ltimer -laabb -o image.run -D TYPE=0

image0.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -lvector3 -lray -lhit -lbasiccamera -ltimer -laabb -o image.run -D TYPE=0

image1.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -lvector3 -lray -lhit -lbasiccamera -ltimer -laabb -o image.run -D TYPE=1

image2.run: image.cpp
	g++ -I ../../include image.cpp -L ../../lib -lvector3 -lray -lhit -lbasiccamera -ltimer -laabb -o image.run -D TYPE=2

clean:
	rm image.run
