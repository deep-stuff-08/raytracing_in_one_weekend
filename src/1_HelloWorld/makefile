.PHONY: all cpu cuda clean

LIB_PATH=../../lib

all: run

cpu: cpu.run
	./cpu.run
	xdg-open outputcpu.png

cuda: cuda.run
	./cuda.run
	xdg-open outputcuda.png

libcpuraytracer.so:
	cd ../../lib && $(MAKE) $@

cpu.run: cpu.cpp libcpuraytracer.so
	g++ -I ../../include cpu.cpp -L $(LIB_PATH) -Wl,-rpath=$(LIB_PATH) -lcpuraytracer -o cpu.run

cuda.run: cuda.cu
	nvcc -I ../../include cuda.cu -o cuda.run

clean:
	rm cpu.run cuda.run
