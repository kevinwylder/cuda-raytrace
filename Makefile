
build: main.cu math.h util.c image.h
	nvcc -g -O0 -G main.cu -o build -lnvjpeg

run: build
	./build

debug: build
	cuda-gdb ./build

clean:
	rm build
