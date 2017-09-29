CFLAGS=-std=gnu99 -Wall -g -O2
LDLIBS=-lm -lrt 
#-lOpenCL

all: seq cuda 

seq: main.cu cnn_seq.cu timer.cu	
	nvcc -o cnn_seq main.cu cnn_seq.cu timer.cu
#cnn_seq: main.cu timer.cu

cuda: main.cu cnn_cuda.cu timer.cu
	nvcc -o cnn_cuda main.cu cnn_cuda.cu timer.cu

cuda_mul: main.cu cnn_cuda_mul.cu timer.cu
	nvcc -o cnn_cuda_mul main.cu cnn_cuda_mul.cu timer.cu

debug: main.cu cnn_cuda_mul.cu timer.cu
	nvcc -g -G -o cnn_cuda_mul main.cu cnn_cuda_mul.cu timer.cu
#cnn_opencl: main.o timer.o

clean:
	rm -f main.o timer.o cnn_seq cnn_cuda cnn_cuda_mul result*.out
