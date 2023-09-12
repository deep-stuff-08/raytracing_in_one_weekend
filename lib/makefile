.PHONY: clean

SRCS := $(wildcard src/*.cpp)
OBJS := $(patsubst src/%.cpp, bin/%.o, $(SRCS))
all: libcpuraytracer.so

libcpuraytracer.so: bin $(OBJS)
	g++ $(OBJS) -o libcpuraytracer.so -shared

bin:
	mkdir bin

bin/%.o : src/%.cpp
	g++ -fPIC -I ../include -c $^ -o $@

clean:
	rm -f $(OBJS)