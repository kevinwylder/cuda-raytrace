
build: main.cu scene.cu scene.h
	nvcc -g -O0 -G main.cu -o build -lnvjpeg

run: build
	./build

debug: build
	cuda-gdb ./build

clean:
	rm build
