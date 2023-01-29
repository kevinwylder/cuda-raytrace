
build: main.c
	nvcc -g -O0 -G main.c -o build -lnvjpeg

run: build
	./build

debug: build
	cuda-gdb ./build

clean:
	rm build
