.PHONY: run clean

LIB_PATH=../../lib

all: run

run: image.run
	./image.run
	xdg-open output.png

libcpuraytracer.so:
	cd ../../lib && $(MAKE) $@

image.run: image.cpp libcpuraytracer.so
	g++ -I ../../include image.cpp -L $(LIB_PATH) -Wl,-rpath=$(LIB_PATH) -lcpuraytracer -o image.run

clean:
	rm image.run
