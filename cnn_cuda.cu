#include "cnn.h"
#include "timer.h"
#include <thrust/device_vector.h>
#include <stdio.h>

/*
 * TODO
 * Define kernel here
 */
__global__
void pooling(
   float * inputs,
   float * outputs,
   int N,
   int D)
{
   // Store each work-item’s unique row and column
   int i = blockIdx.x * blockDim.x + threadIdx.x; // N*N
   int j = blockIdx.y * blockDim.y + threadIdx.y; // D

   if (i < N*N && j < D) {
       int x = i/N; int y = i - x*N;
       float max = 0;
       for (int k = 0; k < 2; k++) {
         for (int l = 0; l < 2; l++) {
            float pixel = inputs[(x*2 + k)*2*N + y*2+l + (j*N*N*4)];
            max = (max > pixel) ? max : pixel;
         }
       }
       outputs[i + (j*N*N)] = max;
   }
}

__global__
void convolution_v2(
   float * inputs,
   float * outputs,
   float * filters,
   float * biases,
   int N,
   int D1, 
   int D2)
{
    // Store each work-item’s unique row and column
   int d = blockIdx.x * blockDim.x + threadIdx.x; // N*N
   int d1 = blockIdx.y * blockDim.y + threadIdx.y; // D1
   int z = blockIdx.z * blockDim.z + threadIdx.z; // D2

   if (d < N*N && d1 < D1 && z < D2) {
        int i = d/N; int j = d- i*N;
        int oIdx = i*N + j + (N*N*z);
        outputs[oIdx] = 0;
        
        float sum = 0;
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < 3; l++) {
                int x = i + k - 1;
                int y = j + l - 1;
                if (x >= 0 && x < N && y >= 0 && y < N)
                    sum += inputs[x*N + y + N*N*d1] * filters[k*3 + l + (3*3 * (z*D1 + d1))];
            }
        }
        atomicAdd(&outputs[oIdx], sum);
        __syncthreads();
        // RELU
        float bias = biases[z];
        outputs[oIdx] = (outputs[oIdx] + bias > 0) ? (outputs[oIdx] + bias) : 0;
    }
}

__global__
void convolution_v1(
   float * inputs,
   float * outputs,
   float * filters,
   float * biases,
   int N,
   int D1, 
   int D2)
{
   // Store each work-item’s unique row and column
   int d = blockIdx.x * blockDim.x + threadIdx.x; // N*N
   int z = blockIdx.y * blockDim.y + threadIdx.y; // D2

   if (d < N*N && z < D2) {
        int i = d/N; int j = d- i*N;
        int oIdx = i*N + j + (N*N*z);
        outputs[oIdx] = 0;
        // Unroll 4 times
        for (int t = D1 % 4; t < D1; t+=4) {
            float sum = 0;
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    int x = i + k - 1;
                    int y = j + l - 1;
                    if (x >= 0 && x < N && y >= 0 && y < N)
                        sum += inputs[x*N + y + N*N*t] * filters[k*3 + l + (3*3 * (z*D1 + t))];
                }
            }
            outputs[oIdx] += sum;

            sum = 0;
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    int x = i + k - 1;
                    int y = j + l - 1;
                    if (x >= 0 && x < N && y >= 0 && y < N)
                        sum += inputs[x*N + y + N*N*(t+1)] * filters[k*3 + l + (3*3 * (z*D1 + (t+1)))];
                }
            }
            outputs[oIdx] += sum;

            sum = 0;
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    int x = i + k - 1;
                    int y = j + l - 1;
                    if (x >= 0 && x < N && y >= 0 && y < N)
                        sum += inputs[x*N + y + N*N*(t+2)] * filters[k*3 + l + (3*3 * (z*D1 + (t+2)))];
                }
            }
            outputs[oIdx] += sum;

            sum = 0;
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    int x = i + k - 1;
                    int y = j + l - 1;
                    if (x >= 0 && x < N && y >= 0 && y < N)
                        sum += inputs[x*N + y + N*N*(t+3)] * filters[k*3 + l + (3*3 * (z*D1 + (t+3)))];
                }
            }
            outputs[oIdx] += sum;
        }
        for (int t = 0; t < D1 % 4; t++) {
            float sum = 0;
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    int x = i + k - 1;
                    int y = j + l - 1;
                    if (x >= 0 && x < N && y >= 0 && y < N)
                        sum += inputs[x*N + y + N*N*t] * filters[k*3 + l + (3*3 * (z*D1 + t))];
                }
            }
            outputs[oIdx] += sum;
        }
        // RELU
        float bias = biases[z];
        outputs[oIdx] = (outputs[oIdx] + bias > 0) ? (outputs[oIdx] + bias) : 0;
    }
}


__global__
void fc(
   float * input_neuron,
   float * output_neuron,
   const int N,
   const int M,
   float * weights,
   float * biases)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;

   if (x < M) {
       float sum = 0;

       for (int i = 0; i < N; i++) {
          sum += weights[x*N + i] * input_neuron[i];
       }
       output_neuron[x] = sum + biases[x];
       // RELU
       if (output_neuron[x] < 0) {
          output_neuron[x] = 0;
       }
   }
}

__global__
void softmax_kernel(
    float * output,
    int N)
{
    int i = threadIdx.x;
    float sum = 0;
    for (i = 0; i < N; i++) {
        sum += exp(output[i]);
    }
    for (i = 0; i < N; i++) {
        output[i] = exp(output[i]) / sum;
    }
}


/************************   CUDA   ************************/

#define NormalToOne(x) (((x)<=0)?(1):x)

float data_transfer_time = 0;
float pooling_time = 0;
float conv_time = 0;
float fc_time = 0;
float softmax_time = 0;

/*
 * D = channel size
 * N = width and height of an output image
 * Thus, input is (D, N * 2, N * 2) and output is (D, N, N).
 */
static void pooling_layer(float *inputs, float *outputs, int D, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Call gpu kernel
    dim3 threadsPerBlock(32, 32, 1);
    if (N < 4) threadsPerBlock.x = N*N;
    threadsPerBlock.y = NormalToOne(1024 / threadsPerBlock.x);
    dim3 numBlocks((N*N + threadsPerBlock.x - 1)/threadsPerBlock.x, 
                    (D + threadsPerBlock.y - 1)/threadsPerBlock.y,
                    1);

    cudaEventRecord(start);
    pooling<<<numBlocks, threadsPerBlock>>>(inputs, outputs, N, D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    pooling_time += milliseconds;
}

static void convolution_layer_v2(float *inputs, float *outputs, float *filters, float *biases, int D2, int D1, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Call GPU kernel
    dim3 threadsPerBlock(8, 8, 16);
    if (N < 4) threadsPerBlock.x = N*N;
    threadsPerBlock.y = NormalToOne(1024 / (threadsPerBlock.x * threadsPerBlock.z));
    dim3 numBlocks((N*N + threadsPerBlock.x - 1)/threadsPerBlock.x, 
                    (D1 + threadsPerBlock.y - 1)/threadsPerBlock.y, 
                    (D2 + threadsPerBlock.z - 1)/threadsPerBlock.z);

    cudaEventRecord(start);
    convolution_v2<<<numBlocks, threadsPerBlock>>>(inputs, outputs, filters, biases, N, D1, D2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    conv_time += milliseconds;
}

/*
 * D2 = output channel size
 * D1 = input channel size
 * N = width and height of an input image
 * input image is zero-padded by 1.
 * Thus, input is (D1, N, N) and output is (D2, N, N)
 */
static void convolution_layer_v1(float *inputs, float *outputs, float *filters, float *biases, int D2, int D1, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Call GPU kernel
    dim3 threadsPerBlock(16, 32, 1);
    if (N < 4) threadsPerBlock.x = N*N;
    threadsPerBlock.y = NormalToOne(512 / threadsPerBlock.x);
    dim3 numBlocks((N*N + threadsPerBlock.x - 1)/threadsPerBlock.x, 
                    (D2 + threadsPerBlock.y - 1)/threadsPerBlock.y, 
                    1);

    cudaEventRecord(start);
    convolution_v1<<<numBlocks, threadsPerBlock>>>(inputs, outputs, filters, biases, N, D1, D2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    conv_time += milliseconds;
}


/*
 * M = output size
 * N = input size
 */
static void fc_layer(float *input_neuron, float *output_neuron, float *weights, float *biases, int M, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Call GPU kernel
    int blockSize = 512;
    if (M < 512) blockSize = M;
    int gridSize = (M + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    fc<<<gridSize, blockSize>>>(input_neuron, output_neuron, N, M, weights, biases);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    fc_time += milliseconds;
}


static void softmax(float *output, int N) {
    timer_start(1);

    int i;
    float max = output[0];
    for (i = 1; i < N; i++) {
        max = (output[i] > max)?output[i]:max;
    }
    float sum = 0;
    for (i = 0; i < N; i++) {
        sum += exp(output[i] - max);
    }
    for (i = 0; i < N; i++) {
        output[i] = exp(output[i] - max) / sum;
    }
    softmax_time += timer_end(1)*1000;
}

static int find_max(float *fc, int N) {
    int i;
    int maxid = 0;
    float maxval = 0;
    for (i = 0; i < N; i++) {
        if (maxval < fc[i]) {
            maxval = fc[i];
            maxid = i;
        }
    }
    return maxid;
}

float* alloc_layer(size_t n) {
    return (float*)malloc(n * sizeof(float));
}


void cnn_init() {
    /*
     * TODO
     * Initialize OpenCL objects as global variables. For example,
     * clGetPlatformIDs(1, &platform, NULL);
     */
}

const int NETWORK_SIZES[] = {
    64 * 3 * 3 * 3, 64,
    64 * 64 * 3 * 3, 64,
    128 * 64 * 3 * 3, 128,
    128 * 128 * 3 * 3, 128,
    256 * 128 * 3 * 3, 256,
    256 * 256 * 3 * 3, 256,
    256 * 256 * 3 * 3, 256,
    512 * 256 * 3 * 3, 512,
    512 * 512 * 3 * 3, 512,
    512 * 512 * 3 * 3, 512,
    512 * 512 * 3 * 3, 512,
    512 * 512 * 3 * 3, 512,
    512 * 512 * 3 * 3, 512,
    512 * 512, 512,
    512 * 512, 512,
    10 * 512, 10
};

const int OUTPUT_SIZES[] = {
    64 * 32 * 32,
    64 * 32 * 32,
    64 * 16 * 16,
    128 * 16 * 16,
    128 * 16 * 16,
    128 * 8 * 8,
    256 * 8 * 8,
    256 * 8 * 8,
    256 * 8 * 8,
    256 * 4 * 4,
    512 * 4 * 4,
    512 * 4 * 4,
    512 * 4 * 4,
    512 * 2 * 2,
    512 * 2 * 2,
    512 * 2 * 2,
    512 * 2 * 2,
    512 * 1 * 1,
    512,
    512,
    10
};

void print_matrix(float *matrix, int size, const char *desc) {
    int matrix_dense = 0;
    int matrix_zero = 0;
    for (int i = 0; i < size; i++)
	    if (matrix[i] > 0) matrix_dense++;
	    else matrix_zero++;
    //printf("%s - dense ratio = %.3f\n", desc, ((float)matrix_dense/(matrix_dense + matrix_zero)));
}

void compare_matrix(float *matrix1, float *matrix2, int size, const char *desc) {
    int matrix_equal = 0;
    for (int i = 0; i < size; i++) {
	    if (matrix1[i] == matrix2[i]) matrix_equal++;
    }
    printf("%s - equal ratio = %.3f\n", desc, ((float)matrix_equal/size));
}

void cnn(float *images, float **network, int *labels, float *confidences, int num_images, int batch_size) {
    /*
     * TODO
     * Implement here.
     * Write classification results to labels and confidences.
     * See "cnn_seq.c" if you don't know what to do.
     */
    // slice the network into weights and biases
    float *w1_1, *b1_1, *w1_2, *b1_2;
    float *w2_1, *b2_1, *w2_2, *b2_2;
    float *w3_1, *b3_1, *w3_2, *b3_2, *w3_3, *b3_3;
    float *w4_1, *b4_1, *w4_2, *b4_2, *w4_3, *b4_3;
    float *w5_1, *b5_1, *w5_2, *b5_2, *w5_3, *b5_3;
    float *w1, *b1, *w2, *b2, *w3, *b3;

    // Set data for weights and biases
    w1_1 = network[0]; b1_1 = network[1];
    w1_2 = network[2]; b1_2 = network[3];
    w2_1 = network[4]; b2_1 = network[5];
    w2_2 = network[6]; b2_2 = network[7];
    w3_1 = network[8]; b3_1 = network[9];
    w3_2 = network[10]; b3_2 = network[11];
    w3_3 = network[12]; b3_3 = network[13];
    w4_1 = network[14]; b4_1 = network[15];
    w4_2 = network[16]; b4_2 = network[17];
    w4_3 = network[18]; b4_3 = network[19];
    w5_1 = network[20]; b5_1 = network[21];
    w5_2 = network[22]; b5_2 = network[23];
    w5_3 = network[24]; b5_3 = network[25];
    w1 = network[26]; b1 = network[27];
    w2 = network[28]; b2 = network[29];
    w3 = network[30]; b3 = network[31];

    // Allocate vectors in device memory
    float *d_w1_1, *d_b1_1, *d_w1_2, *d_b1_2;
    float *d_w2_1, *d_b2_1, *d_w2_2, *d_b2_2;
    float *d_w3_1, *d_b3_1, *d_w3_2, *d_b3_2, *d_w3_3, *d_b3_3;
    float *d_w4_1, *d_b4_1, *d_w4_2, *d_b4_2, *d_w4_3, *d_b4_3;
    float *d_w5_1, *d_b5_1, *d_w5_2, *d_b5_2, *d_w5_3, *d_b5_3;
    float *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3;

    cudaMalloc(&d_w1_1, NETWORK_SIZES[0] * sizeof(float));
    cudaMalloc(&d_w1_2, NETWORK_SIZES[2] * sizeof(float));
    cudaMalloc(&d_w2_1, NETWORK_SIZES[4] * sizeof(float));
    cudaMalloc(&d_w2_2, NETWORK_SIZES[6] * sizeof(float));
    cudaMalloc(&d_w3_1, NETWORK_SIZES[8] * sizeof(float));
    cudaMalloc(&d_w3_2, NETWORK_SIZES[10] * sizeof(float));
    cudaMalloc(&d_w3_3, NETWORK_SIZES[12] * sizeof(float));
    cudaMalloc(&d_w4_1, NETWORK_SIZES[14] * sizeof(float));
    cudaMalloc(&d_w4_2, NETWORK_SIZES[16] * sizeof(float));
    cudaMalloc(&d_w4_3, NETWORK_SIZES[18] * sizeof(float));
    cudaMalloc(&d_w5_1, NETWORK_SIZES[20] * sizeof(float));
    cudaMalloc(&d_w5_2, NETWORK_SIZES[22] * sizeof(float));
    cudaMalloc(&d_w5_3, NETWORK_SIZES[24] * sizeof(float));
    cudaMalloc(&d_w1, NETWORK_SIZES[26] * sizeof(float));
    cudaMalloc(&d_w2, NETWORK_SIZES[28] * sizeof(float));
    cudaMalloc(&d_w3, NETWORK_SIZES[30] * sizeof(float));

    cudaMalloc(&d_b1_1, NETWORK_SIZES[1] * sizeof(float));
    cudaMalloc(&d_b1_2, NETWORK_SIZES[3] * sizeof(float));
    cudaMalloc(&d_b2_1, NETWORK_SIZES[5] * sizeof(float));
    cudaMalloc(&d_b2_2, NETWORK_SIZES[7] * sizeof(float));
    cudaMalloc(&d_b3_1, NETWORK_SIZES[9] * sizeof(float));
    cudaMalloc(&d_b3_2, NETWORK_SIZES[11] * sizeof(float));
    cudaMalloc(&d_b3_3, NETWORK_SIZES[13] * sizeof(float));
    cudaMalloc(&d_b4_1, NETWORK_SIZES[15] * sizeof(float));
    cudaMalloc(&d_b4_2, NETWORK_SIZES[17] * sizeof(float));
    cudaMalloc(&d_b4_3, NETWORK_SIZES[19] * sizeof(float));
    cudaMalloc(&d_b5_1, NETWORK_SIZES[21] * sizeof(float));
    cudaMalloc(&d_b5_2, NETWORK_SIZES[23] * sizeof(float));
    cudaMalloc(&d_b5_3, NETWORK_SIZES[25] * sizeof(float));
    cudaMalloc(&d_b1, NETWORK_SIZES[27] * sizeof(float));
    cudaMalloc(&d_b2, NETWORK_SIZES[29] * sizeof(float));
    cudaMalloc(&d_b3, NETWORK_SIZES[31] * sizeof(float));

    // Create cudaEvent to measure cuda time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy vectors from host memory to device memory
    cudaEventRecord(start);
    cudaMemcpy(d_w1_1, w1_1, NETWORK_SIZES[0] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1_2, w1_2, NETWORK_SIZES[2] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2_1, w2_1, NETWORK_SIZES[4] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2_2, w2_2, NETWORK_SIZES[6] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3_1, w3_1, NETWORK_SIZES[8] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3_2, w3_2, NETWORK_SIZES[10] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3_3, w3_3, NETWORK_SIZES[12] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w4_1, w4_1, NETWORK_SIZES[14] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w4_2, w4_2, NETWORK_SIZES[16] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w4_3, w4_3, NETWORK_SIZES[18] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w5_1, w5_1, NETWORK_SIZES[20] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w5_2, w5_2, NETWORK_SIZES[22] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w5_3, w5_3, NETWORK_SIZES[24] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1, w1, NETWORK_SIZES[26] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, w2, NETWORK_SIZES[28] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, w3, NETWORK_SIZES[30] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1_1, b1_1, NETWORK_SIZES[1] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1_2, b1_2, NETWORK_SIZES[3] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2_1, b2_1, NETWORK_SIZES[5] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2_2, b2_2, NETWORK_SIZES[7] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3_1, b3_1, NETWORK_SIZES[9] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3_2, b3_2, NETWORK_SIZES[11] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3_3, b3_3, NETWORK_SIZES[13] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b4_1, b4_1, NETWORK_SIZES[15] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b4_2, b4_2, NETWORK_SIZES[17] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b4_3, b4_3, NETWORK_SIZES[19] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b5_1, b5_1, NETWORK_SIZES[21] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b5_2, b5_2, NETWORK_SIZES[23] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b5_3, b5_3, NETWORK_SIZES[25] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1, NETWORK_SIZES[27] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2, NETWORK_SIZES[29] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, b3, NETWORK_SIZES[31] * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    data_transfer_time = milliseconds;
    printf("network data transfer time = %f ms\n", data_transfer_time);
    data_transfer_time = 0;

    // allocate memory for output of each layer
    float *c1_1, *c1_2, *p1;
    float *c2_1, *c2_2, *p2;
    float *c3_1, *c3_2, *c3_3, *p3;
    float *c4_1, *c4_2, *c4_3, *p4;
    float *c5_1, *c5_2, *c5_3, *p5;
    float *fc1, *fc2, *fc3;
    c1_1 = alloc_layer(64 * 32 * 32);
    c1_2 = alloc_layer(64 * 32 * 32);
    p1   = alloc_layer(64 * 16 * 16);
    c2_1 = alloc_layer(128 * 16 * 16);
    c2_2 = alloc_layer(128 * 16 * 16);
    p2   = alloc_layer(128 * 8 * 8);
    c3_1 = alloc_layer(256 * 8 * 8);
    c3_2 = alloc_layer(256 * 8 * 8);
    c3_3 = alloc_layer(256 * 8 * 8);
    p3   = alloc_layer(256 * 4 * 4);
    c4_1 = alloc_layer(512 * 4 * 4);
    c4_2 = alloc_layer(512 * 4 * 4);
    c4_3 = alloc_layer(512 * 4 * 4);
    p4   = alloc_layer(512 * 2 * 2);
    c5_1 = alloc_layer(512 * 2 * 2);
    c5_2 = alloc_layer(512 * 2 * 2);
    c5_3 = alloc_layer(512 * 2 * 2);
    p5   = alloc_layer(512 * 1 * 1);
    fc1  = alloc_layer(512);
    fc2  = alloc_layer(512);
    fc3  = alloc_layer(10);

    float *c1_1_seq, *c1_2_seq, *p1_seq;
    float *c2_1_seq, *c2_2_seq, *p2_seq;
    float *c3_1_seq, *c3_2_seq, *c3_3_seq, *p3_seq;
    float *c4_1_seq, *c4_2_seq, *c4_3_seq, *p4_seq;
    float *c5_1_seq, *c5_2_seq, *c5_3_seq, *p5_seq;
    float *fc1_seq, *fc2_seq, *fc3_seq;
    c1_1_seq = alloc_layer(64 * 32 * 32);
    c1_2_seq = alloc_layer(64 * 32 * 32);
    p1_seq   = alloc_layer(64 * 16 * 16);
    c2_1_seq = alloc_layer(128 * 16 * 16);
    c2_2_seq = alloc_layer(128 * 16 * 16);
    p2_seq   = alloc_layer(128 * 8 * 8);
    c3_1_seq = alloc_layer(256 * 8 * 8);
    c3_2_seq = alloc_layer(256 * 8 * 8);
    c3_3_seq = alloc_layer(256 * 8 * 8);
    p3_seq   = alloc_layer(256 * 4 * 4);
    c4_1_seq = alloc_layer(512 * 4 * 4);
    c4_2_seq = alloc_layer(512 * 4 * 4);
    c4_3_seq = alloc_layer(512 * 4 * 4);
    p4_seq   = alloc_layer(512 * 2 * 2);
    c5_1_seq = alloc_layer(512 * 2 * 2);
    c5_2_seq = alloc_layer(512 * 2 * 2);
    c5_3_seq = alloc_layer(512 * 2 * 2);
    p5_seq   = alloc_layer(512 * 1 * 1);
    fc1_seq  = alloc_layer(512);
    fc2_seq  = alloc_layer(512);
    fc3_seq = alloc_layer(10);
    
    // Allocate output vectors in device memory to transfer between layers
    float *d_c1_1, *d_c1_2, *d_p1;
    float *d_c2_1, *d_c2_2, *d_p2;
    float *d_c3_1, *d_c3_2, *d_c3_3, *d_p3;
    float *d_c4_1, *d_c4_2, *d_c4_3, *d_p4;
    float *d_c5_1, *d_c5_2, *d_c5_3, *d_p5;
    float *d_fc1, *d_fc2, *d_fc3;
    cudaMalloc(&d_c1_1, OUTPUT_SIZES[0] * sizeof(float));
    cudaMalloc(&d_c1_2, OUTPUT_SIZES[1] * sizeof(float));
    cudaMalloc(&d_p1,   OUTPUT_SIZES[2] * sizeof(float));
    cudaMalloc(&d_c2_1, OUTPUT_SIZES[3] * sizeof(float));
    cudaMalloc(&d_c2_2, OUTPUT_SIZES[4] * sizeof(float));
    cudaMalloc(&d_p2,   OUTPUT_SIZES[5] * sizeof(float));
    cudaMalloc(&d_c3_1, OUTPUT_SIZES[6] * sizeof(float));
    cudaMalloc(&d_c3_2, OUTPUT_SIZES[7] * sizeof(float));
    cudaMalloc(&d_c3_3, OUTPUT_SIZES[8] * sizeof(float));
    cudaMalloc(&d_p3,   OUTPUT_SIZES[9] * sizeof(float));
    cudaMalloc(&d_c4_1, OUTPUT_SIZES[10] * sizeof(float));
    cudaMalloc(&d_c4_2, OUTPUT_SIZES[11] * sizeof(float));
    cudaMalloc(&d_c4_3, OUTPUT_SIZES[12] * sizeof(float));
    cudaMalloc(&d_p4,   OUTPUT_SIZES[13] * sizeof(float));
    cudaMalloc(&d_c5_1, OUTPUT_SIZES[14] * sizeof(float));
    cudaMalloc(&d_c5_2, OUTPUT_SIZES[15] * sizeof(float));
    cudaMalloc(&d_c5_3, OUTPUT_SIZES[16] * sizeof(float));
    cudaMalloc(&d_p5,   OUTPUT_SIZES[17] * sizeof(float));
    cudaMalloc(&d_fc1,  OUTPUT_SIZES[18] * sizeof(float));
    cudaMalloc(&d_fc2,  OUTPUT_SIZES[19] * sizeof(float));
    cudaMalloc(&d_fc3,  OUTPUT_SIZES[20] * sizeof(float));
    

    // run network
    size_t image_size = 3*32*32 * sizeof(float);
    float *d_image;
    cudaMalloc(&d_image, image_size);
    for(int i = 0; i < num_images; i++)
    {
        // Copy image from host to device
        float *image = images + i * 3 * 32 * 32;
        print_matrix(image, 3*32*32, "image");

        cudaEventRecord(start);
        cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        data_transfer_time += milliseconds;

        convolution_layer_v1(d_image, d_c1_1, d_w1_1, d_b1_1, 64, 3, 32);
        convolution_layer_v1(d_c1_1, d_c1_2, d_w1_2, d_b1_2, 64, 64, 32);
        pooling_layer(d_c1_2, d_p1, 64, 16);

        convolution_layer_v1(d_p1, d_c2_1, d_w2_1, d_b2_1, 128, 64, 16);
        convolution_layer_v1(d_c2_1, d_c2_2, d_w2_2, d_b2_2, 128, 128, 16);
        pooling_layer(d_c2_2, d_p2, 128, 8);

        convolution_layer_v1(d_p2, d_c3_1, d_w3_1, d_b3_1, 256, 128, 8);
        convolution_layer_v1(d_c3_1, d_c3_2, d_w3_2, d_b3_2, 256, 256, 8);
        convolution_layer_v1(d_c3_2, d_c3_3, d_w3_3, d_b3_3, 256, 256, 8);
        pooling_layer(d_c3_3, d_p3, 256, 4);

        convolution_layer_v1(d_p3, d_c4_1, d_w4_1, d_b4_1, 512, 256, 4);
        convolution_layer_v1(d_c4_1, d_c4_2, d_w4_2, d_b4_2, 512, 512, 4);
        convolution_layer_v1(d_c4_2, d_c4_3, d_w4_3, d_b4_3, 512, 512, 4);
        pooling_layer(d_c4_3, d_p4, 512, 2);

        convolution_layer_v1(d_p4, d_c5_1, d_w5_1, d_b5_1, 512, 512, 2);
        convolution_layer_v1(d_c5_1, d_c5_2, d_w5_2, d_b5_2, 512, 512, 2);
        convolution_layer_v1(d_c5_2, d_c5_3, d_w5_3, d_b5_3, 512, 512, 2);
        pooling_layer(d_c5_3, d_p5, 512, 1);

        fc_layer(d_p5, d_fc1, d_w1, d_b1, 512, 512);
        fc_layer(d_fc1, d_fc2, d_w2, d_b2, 512, 512);
        fc_layer(d_fc2, d_fc3, d_w3, d_b3, 10, 512);

        // Copy result from device memory to host memory
        cudaEventRecord(start);
        cudaMemcpy(fc3, d_fc3, OUTPUT_SIZES[20] * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        data_transfer_time += milliseconds;

        // Get the predicted label
        softmax(fc3, 10);        
        labels[i] = find_max(fc3, 10);
        confidences[i] = fc3[labels[i]];
        printf("Image: %d, confidences = %.3f \n", labels[i], confidences[i]);

	// print matrix dense
	cudaMemcpy(c1_1, d_c1_1, OUTPUT_SIZES[0] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(c1_1, OUTPUT_SIZES[0], "c1_1");
        cudaMemcpy(c1_2, d_c1_2, OUTPUT_SIZES[1] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(c1_2, OUTPUT_SIZES[1], "c1_2");
        cudaMemcpy(p1, d_p1, OUTPUT_SIZES[2] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(p1, OUTPUT_SIZES[2], "p1");
        cudaMemcpy(c2_1, d_c2_1, OUTPUT_SIZES[3] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(c2_1, OUTPUT_SIZES[3], "c2_1");
        cudaMemcpy(c2_2, d_c2_2, OUTPUT_SIZES[4] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(c2_2, OUTPUT_SIZES[4], "c2_2");
        cudaMemcpy(p2, d_p2, OUTPUT_SIZES[5] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(p2, OUTPUT_SIZES[5], "p2");
        cudaMemcpy(c3_1, d_c3_1, OUTPUT_SIZES[6] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(c3_1, OUTPUT_SIZES[6], "c3_1");
        cudaMemcpy(c3_2, d_c3_2, OUTPUT_SIZES[7] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(c3_2, OUTPUT_SIZES[7], "c3_2");
        cudaMemcpy(c3_3, d_c3_3, OUTPUT_SIZES[8] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(c3_3, OUTPUT_SIZES[8], "c3_3");
        cudaMemcpy(p3, d_p3, OUTPUT_SIZES[9] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(p3, OUTPUT_SIZES[9], "p3");
        cudaMemcpy(c4_1, d_c4_1, OUTPUT_SIZES[10] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(c4_1, OUTPUT_SIZES[10], "c4_1");
        cudaMemcpy(c4_2, d_c4_2, OUTPUT_SIZES[11] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(c4_2, OUTPUT_SIZES[11], "c4_2");
        cudaMemcpy(c4_3, d_c4_3, OUTPUT_SIZES[12] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(c4_3, OUTPUT_SIZES[12], "c4_3");
        cudaMemcpy(p4, d_p4, OUTPUT_SIZES[13] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(p4, OUTPUT_SIZES[13], "p4");
        cudaMemcpy(c5_1, d_c5_1, OUTPUT_SIZES[14] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(c5_1, OUTPUT_SIZES[14], "c5_1");
        cudaMemcpy(c5_2, d_c5_2, OUTPUT_SIZES[15] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(c5_2, OUTPUT_SIZES[15], "c5_2");
        cudaMemcpy(c5_3, d_c5_3, OUTPUT_SIZES[16] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(c5_3, OUTPUT_SIZES[16], "c5_3");
        cudaMemcpy(p5, d_p5, OUTPUT_SIZES[17] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(p5, OUTPUT_SIZES[17], "p5");
        cudaMemcpy(fc1, d_fc1, OUTPUT_SIZES[18] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(fc1, OUTPUT_SIZES[18], "fc1");
        cudaMemcpy(fc2, d_fc2, OUTPUT_SIZES[19] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(fc2, OUTPUT_SIZES[19], "fc2");
        cudaMemcpy(fc3, d_fc3, OUTPUT_SIZES[20] * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(fc3, OUTPUT_SIZES[20], "fc3");

	/*************************/
	// Copy image from host to device
        float *image1 = images + (i+1) * 3 * 32 * 32;
        compare_matrix(image, image1, 3*32*32, "image");

        cudaEventRecord(start);
        cudaMemcpy(d_image, image1, image_size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        data_transfer_time += milliseconds;

        convolution_layer_v1(d_image, d_c1_1, d_w1_1, d_b1_1, 64, 3, 32);
        convolution_layer_v1(d_c1_1, d_c1_2, d_w1_2, d_b1_2, 64, 64, 32);
        pooling_layer(d_c1_2, d_p1, 64, 16);

        convolution_layer_v1(d_p1, d_c2_1, d_w2_1, d_b2_1, 128, 64, 16);
        convolution_layer_v1(d_c2_1, d_c2_2, d_w2_2, d_b2_2, 128, 128, 16);
        pooling_layer(d_c2_2, d_p2, 128, 8);

        convolution_layer_v1(d_p2, d_c3_1, d_w3_1, d_b3_1, 256, 128, 8);
        convolution_layer_v1(d_c3_1, d_c3_2, d_w3_2, d_b3_2, 256, 256, 8);
        convolution_layer_v1(d_c3_2, d_c3_3, d_w3_3, d_b3_3, 256, 256, 8);
        pooling_layer(d_c3_3, d_p3, 256, 4);

        convolution_layer_v1(d_p3, d_c4_1, d_w4_1, d_b4_1, 512, 256, 4);
        convolution_layer_v1(d_c4_1, d_c4_2, d_w4_2, d_b4_2, 512, 512, 4);
        convolution_layer_v1(d_c4_2, d_c4_3, d_w4_3, d_b4_3, 512, 512, 4);
        pooling_layer(d_c4_3, d_p4, 512, 2);

        convolution_layer_v1(d_p4, d_c5_1, d_w5_1, d_b5_1, 512, 512, 2);
        convolution_layer_v1(d_c5_1, d_c5_2, d_w5_2, d_b5_2, 512, 512, 2);
        convolution_layer_v1(d_c5_2, d_c5_3, d_w5_3, d_b5_3, 512, 512, 2);
        pooling_layer(d_c5_3, d_p5, 512, 1);

        fc_layer(d_p5, d_fc1, d_w1, d_b1, 512, 512);
        fc_layer(d_fc1, d_fc2, d_w2, d_b2, 512, 512);
        fc_layer(d_fc2, d_fc3, d_w3, d_b3, 10, 512);

        // Copy result from device memory to host memory
        cudaEventRecord(start);
        cudaMemcpy(fc3, d_fc3, OUTPUT_SIZES[20] * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        data_transfer_time += milliseconds;

        // Get the predicted label
        softmax(fc3, 10);        
        labels[i+1] = find_max(fc3, 10);
        confidences[i+1] = fc3[labels[i+1]];
        printf("Image: %d, confidences = %.3f \n", labels[i+1], confidences[i+1]);

	// print matrix dense
	cudaMemcpy(c1_1_seq, d_c1_1, OUTPUT_SIZES[0] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(c1_1, c1_1_seq, OUTPUT_SIZES[0], "c1_1");
        cudaMemcpy(c1_2_seq, d_c1_2, OUTPUT_SIZES[1] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(c1_2, c1_2_seq, OUTPUT_SIZES[1], "c1_2");
        cudaMemcpy(p1_seq, d_p1, OUTPUT_SIZES[2] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(p1, p1_seq, OUTPUT_SIZES[2], "p1");
        cudaMemcpy(c2_1_seq, d_c2_1, OUTPUT_SIZES[3] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(c2_1, c2_1_seq, OUTPUT_SIZES[3], "c2_1");
        cudaMemcpy(c2_2_seq, d_c2_2, OUTPUT_SIZES[4] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(c2_2, c2_2_seq, OUTPUT_SIZES[4], "c2_2");
        cudaMemcpy(p2_seq, d_p2, OUTPUT_SIZES[5] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(p2, p2_seq, OUTPUT_SIZES[5], "p2");
        cudaMemcpy(c3_1_seq, d_c3_1, OUTPUT_SIZES[6] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(c3_1, c3_1_seq, OUTPUT_SIZES[6], "c3_1");
        cudaMemcpy(c3_2_seq, d_c3_2, OUTPUT_SIZES[7] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(c3_2, c3_2_seq, OUTPUT_SIZES[7], "c3_2");
        cudaMemcpy(c3_3_seq, d_c3_3, OUTPUT_SIZES[8] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(c3_3, c3_3_seq, OUTPUT_SIZES[8], "c3_3");
        cudaMemcpy(p3_seq, d_p3, OUTPUT_SIZES[9] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(p3, p3_seq, OUTPUT_SIZES[9], "p3");
        cudaMemcpy(c4_1_seq, d_c4_1, OUTPUT_SIZES[10] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(c4_1, c4_1_seq, OUTPUT_SIZES[10], "c4_1");
        cudaMemcpy(c4_2_seq, d_c4_2, OUTPUT_SIZES[11] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(c4_2, c4_2_seq, OUTPUT_SIZES[11], "c4_2");
        cudaMemcpy(c4_3_seq, d_c4_3, OUTPUT_SIZES[12] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(c4_3, c4_3_seq, OUTPUT_SIZES[12], "c4_3");
        cudaMemcpy(p4_seq, d_p4, OUTPUT_SIZES[13] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(p4, p4_seq, OUTPUT_SIZES[13], "p4");
        cudaMemcpy(c5_1_seq, d_c5_1, OUTPUT_SIZES[14] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(c5_1, c5_1_seq, OUTPUT_SIZES[14], "c5_1");
        cudaMemcpy(c5_2_seq, d_c5_2, OUTPUT_SIZES[15] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(c5_2, c5_2_seq, OUTPUT_SIZES[15], "c5_2");
        cudaMemcpy(c5_3_seq, d_c5_3, OUTPUT_SIZES[16] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(c5_3, c5_3_seq, OUTPUT_SIZES[16], "c5_3");
        cudaMemcpy(p5_seq, d_p5, OUTPUT_SIZES[17] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(p5, p5_seq, OUTPUT_SIZES[17], "p5");
        cudaMemcpy(fc1_seq, d_fc1, OUTPUT_SIZES[18] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(fc1, fc1_seq, OUTPUT_SIZES[18], "fc1");
        cudaMemcpy(fc2_seq, d_fc2, OUTPUT_SIZES[19] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(fc2, fc2_seq, OUTPUT_SIZES[19], "fc2");
        cudaMemcpy(fc3_seq, d_fc3, OUTPUT_SIZES[20] * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrix(fc3, fc3_seq, OUTPUT_SIZES[20], "fc3");
    }
    printf("data transfer time = %f ms\n", data_transfer_time);
    printf("pooing time = %f ms\n", pooling_time);
    printf("convolution time = %f ms\n", conv_time);
    printf("fully connected time = %f ms\n", fc_time);
    printf("softmax time = %f ms\n", softmax_time);

    cudaFree(d_image);
    cudaFree(d_w1_1); cudaFree(d_b1_1); cudaFree(d_w1_2); cudaFree(d_b1_2);
    cudaFree(d_w2_1); cudaFree(d_b2_2); cudaFree(d_w2_2); cudaFree(d_b2_2);
    cudaFree(d_w3_1); cudaFree(d_b3_1); cudaFree(d_w3_2); cudaFree(d_b3_2); cudaFree(d_w3_3); cudaFree(d_b3_3);
    cudaFree(d_w4_1); cudaFree(d_b4_1); cudaFree(d_w4_2); cudaFree(d_b4_2); cudaFree(d_w4_3); cudaFree(d_b4_3);
    cudaFree(d_w5_1); cudaFree(d_b5_1); cudaFree(d_w5_2); cudaFree(d_b5_2); cudaFree(d_w5_3); cudaFree(d_b5_3);
    cudaFree(d_w1);   cudaFree(d_b1);   cudaFree(d_w2);   cudaFree(d_b2);   cudaFree(d_w3);   cudaFree(d_b3);

    cudaFree(d_c1_1); cudaFree(d_c1_2); cudaFree(d_p1);
    cudaFree(d_c2_1); cudaFree(d_c2_2); cudaFree(d_p2);
    cudaFree(d_c3_1); cudaFree(d_c3_2); cudaFree(d_c3_3); cudaFree(d_p3);
    cudaFree(d_c4_1); cudaFree(d_c4_2); cudaFree(d_c4_3); cudaFree(d_p4);
    cudaFree(d_c5_1); cudaFree(d_c5_2); cudaFree(d_c5_3); cudaFree(d_p5);
    cudaFree(d_fc1); cudaFree(d_fc2); cudaFree(d_fc3);

}
