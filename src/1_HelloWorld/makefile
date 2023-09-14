.PHONY: all cpu cuda clean

LIB_PATH=../../lib

all: cpu cuda

cpu: cpu.run
	./cpu.run
	xdg-open outputcpu.png

cuda: cuda.run
	./cuda.run
	xdg-open outputcuda.png

libcpuraytracer.so:
	cd ../../lib && $(MAKE) $@

libcudaraytracer.a:
	cd ../../lib && $(MAKE) $@

cpu.run: cpu.cpp libcpuraytracer.so
	g++ -I ../../include cpu.cpp -L $(LIB_PATH) -Wl,-rpath=$(LIB_PATH) -lcpuraytracer -o cpu.run

cuda.run: cuda.cu libcudaraytracer.a
	nvcc -I ../../include cuda.cu -L $(LIB_PATH) -Xlinker=-rpath=$(LIB_PATH) -lcudaraytracer -o cuda.run

clean:
	rm cpu.run cuda.run
