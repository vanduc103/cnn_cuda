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
    int D,
    int NoImg)
{
    // Store each work-item’s unique row and column
    int i = blockIdx.x * blockDim.x + threadIdx.x; // N*N
    int j = blockIdx.y * blockDim.y + threadIdx.y; // D
    int n = blockIdx.z * blockDim.z + threadIdx.z; // NoImg

    if (i < N*N && j < D && n < NoImg) {
        int x = i/N; int y = i - x*N;
        float max = 0;
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 2; l++) {
                    float pixel = inputs[(x*2 + k)*2*N + y*2+l + (j*N*N*4) + (4*N*N*D*n)];
                    max = (max > pixel) ? max : pixel;
                }
        }
        outputs[i + (j*N*N) + (N*N*D*n)] = max;
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
   int D2,
   int NoImg)
{
   // Store each work-item’s unique row and column
   int d = blockIdx.x * blockDim.x + threadIdx.x; // N*N
   int d2 = blockIdx.y * blockDim.y + threadIdx.y; // D2
   int n = blockIdx.z * blockDim.z + threadIdx.z; // NoImg

   extern __shared__ float tmpFilters[];

   if (d < N*N && d2 < D2 && n < NoImg) {
        for (int t = 0; t < D1; t+=1) {
            for (int i = 0; i < 9; i++) tmpFilters[i + (3*3* (threadIdx.y*D1 + t))] = filters[i + (3*3 * (d2*D1 + t))];
        }
        __syncthreads();

        int i = d/N; int j = d- i*N;
        int oIdx = i*N + j + (N*N*d2) + (N*N*D2*n);
        outputs[oIdx] = 0;

        // Unroll 1 times
        for (int t = 0; t < D1; t+=1) {
            float sum = 0;
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    int x = i + k - 1;
                    int y = j + l - 1;
                    if (x >= 0 && x < N && y >= 0 && y < N)
                        sum += inputs[x*N + y + N*N*t + (N*N*D1*n)] * filters[k*3 + l + (3*3 * (d2*D1 + t))];
                }
            }
            outputs[oIdx] += sum;
        }
        // RELU
        float bias = biases[d2];
        outputs[oIdx] = (outputs[oIdx] + bias > 0) ? (outputs[oIdx] + bias) : 0;
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
   int D2,
   int NoImg)
{
   // Store each work-item’s unique row and column
   int x1 = blockIdx.x * blockDim.x + threadIdx.x; // N*N*D2*NoImg

   if (x1 < N*N*D2*NoImg) {
        // Calculate index values
        int n = x1/(N*N*D2); int tmp1 = x1 - n*(N*N*D2);
        int d2 = tmp1/(N*N); int tmp2 = tmp1 - d2*(N*N);
        int i = tmp2/N;
        int j = tmp2 - i*N;
        
        int oIdx = x1; //i*N + j + (N*N*d2) + (N*N*D2*n);
        outputs[oIdx] = 0;

        // Unroll 1 times
        for (int t = 0; t < D1; t+=1) {
            float sum = 0;
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    int x = i + k - 1;
                    int y = j + l - 1;
                    if (x >= 0 && x < N && y >= 0 && y < N)
                        sum += inputs[x*N + y + N*N*t + (N*N*D1*n)] * filters[k*3 + l + (3*3 * (d2*D1 + t))];
                }
            }
            outputs[oIdx] += sum;
        }
        // RELU
        float bias = biases[d2];
        outputs[oIdx] = (outputs[oIdx] + bias > 0) ? (outputs[oIdx] + bias) : 0;
    }
}

// Fusing 2 convolution layers
__global__
void convolution_v3(
   float * inputs,
   float * outputs,
   float * filters1,
   float * filters2,
   float * biases1,
   float * biases2,
   int N,
   int D1, 
   int D2,
   int NoImg)
{
   // Store each work-item’s unique row and column
   int x1 = blockIdx.x * blockDim.x + threadIdx.x; // x1 is output index

   
}


__global__
void fc(
   float * input_neuron,
   float * output_neuron,
   float * weights,
   float * biases,
   const int N,
   const int M,
   const int NoImg)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x; // M
   int y = blockIdx.y * blockDim.y + threadIdx.y; // NoImg

   if (x < M && y < NoImg) {
       float sum = 0;

       for (int i = 0; i < N; i++) {
          sum += weights[x*N + i] * input_neuron[i + N*y];
       }
       output_neuron[x + M*y] = sum + biases[x];
       // RELU
       if (output_neuron[x + M*y] < 0) {
          output_neuron[x + M*y] = 0;
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
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// show memory usage of GPU
static void show_mem_gpu(const char *info) {
    size_t free_byte ;
    size_t total_byte ;

    gpuErrchk(cudaMemGetInfo( &free_byte, &total_byte )) ;

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;

    printf("%s - GPU memory usage: used = %.3f MB, free = %.3f MB, total = %.3f MB\n",
        info, used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

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
static void pooling_layer(float *inputs, float *outputs, int D, int N, int NoImg) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    show_mem_gpu("Before pooling");
    // Call gpu kernel
    dim3 threadsPerBlock(8, 8, 1);
    if (N < 4) threadsPerBlock.x = N*N;
    threadsPerBlock.z = NormalToOne(1024 / (threadsPerBlock.x*threadsPerBlock.y));
    dim3 numBlocks((N*N + threadsPerBlock.x - 1)/threadsPerBlock.x, 
                    (D + threadsPerBlock.y - 1)/threadsPerBlock.y,
                    (NoImg + threadsPerBlock.z - 1)/threadsPerBlock.z);

    cudaEventRecord(start);
    pooling<<<numBlocks, threadsPerBlock>>>(inputs, outputs, N, D, NoImg);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    show_mem_gpu("After pooling");

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    pooling_time += milliseconds/1000;
}

static void convolution_layer_v1(float *inputs, float *outputs, float *filters, float *biases, int D2, int D1, int N, int NoImg) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Call GPU kernel
    dim3 threadsPerBlock(8, 8, 16);
    if (N < 4) threadsPerBlock.x = N*N;
    threadsPerBlock.z = NormalToOne(1024 / (threadsPerBlock.x*threadsPerBlock.y));
    dim3 numBlocks((N*N + threadsPerBlock.x - 1)/threadsPerBlock.x, 
                    (D2 + threadsPerBlock.y - 1)/threadsPerBlock.y, 
                    (NoImg + threadsPerBlock.z - 1)/threadsPerBlock.z);

    cudaEventRecord(start);
    convolution_v1<<<numBlocks, threadsPerBlock, 3*3*D1*threadsPerBlock.y*sizeof(float)>>>(inputs, outputs, filters, biases, N, D1, D2, NoImg);
    cudaEventRecord(stop);
    gpuErrchk(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    conv_time += milliseconds/1000;
}

static void convolution_layer_v2(float *inputs, float *outputs, float *filters, float *biases, int D2, int D1, int N, int NoImg) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    show_mem_gpu("Before conv");
    // Call GPU kernel
    dim3 threadsPerBlock(1024, 1, 1);
    dim3 numBlocks((N*N*D2*NoImg + threadsPerBlock.x - 1)/threadsPerBlock.x, 1, 1);

    cudaEventRecord(start);
    convolution_v2<<<numBlocks, threadsPerBlock>>>(inputs, outputs, filters, biases, N, D1, D2, NoImg);
    cudaEventRecord(stop);
    gpuErrchk(cudaEventSynchronize(stop));
    show_mem_gpu("After conv");

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    conv_time += milliseconds/1000;
}


/*
 * M = output size
 * N = input size
 */
static void fc_layer(float *input_neuron, float *output_neuron, float *weights, float *biases, int M, int N, int NoImg) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Call GPU kernel
    dim3 blockSize(16, 1);
    if (M < 64) blockSize.x = M;
    blockSize.y = NormalToOne(1024 / blockSize.x);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (NoImg + blockSize.y - 1)/blockSize.y);

    cudaEventRecord(start);
    fc<<<gridSize, blockSize>>>(input_neuron, output_neuron, weights, biases, N, M, NoImg);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    fc_time += milliseconds/1000;
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
    softmax_time += timer_end(1);
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

    // view networks values
    for (int i = 0; i < NETWORK_SIZES[0]; i++) {
        //printf("w1_1[%d] = %f\n", i, w1_1[i]);
    }

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
    data_transfer_time = milliseconds/1000;
    printf("network data transfer time = %f s\n", data_transfer_time);
    data_transfer_time = 0;
    show_mem_gpu("After network data transfer");

    // Batch images size
    int batchImg = batch_size;
    int batchImg2 = 2*batchImg;
    printf("batch size = %d\n", batchImg);

    // Allocate output vectors in device memory to transfer between layers
    /*float *d_c1_1, *d_c1_2, *d_p1;
    float *d_c2_1, *d_c2_2, *d_p2;
    float *d_c3_1, *d_c3_2, *d_c3_3, *d_p3;
    float *d_c4_1, *d_c4_2, *d_c4_3, *d_p4;
    float *d_c5_1, *d_c5_2, *d_c5_3, *d_p5;
    float *d_fc1, *d_fc2, *d_fc3;*/
    float *d1, *d2, *p1;
    cudaMalloc(&d1, batchImg * OUTPUT_SIZES[0] * sizeof(float));
    cudaMalloc(&d2, batchImg * OUTPUT_SIZES[1] * sizeof(float));
    cudaMalloc(&p1, batchImg * OUTPUT_SIZES[2] * sizeof(float));
    /*cudaMalloc(&d_p1,   batchImg * OUTPUT_SIZES[2] * sizeof(float));
    cudaMalloc(&d_c2_1, batchImg * OUTPUT_SIZES[3] * sizeof(float));
    cudaMalloc(&d_c2_2, batchImg * OUTPUT_SIZES[4] * sizeof(float));
    cudaMalloc(&d_p2,   batchImg * OUTPUT_SIZES[5] * sizeof(float));
    cudaMalloc(&d_c3_1, batchImg * OUTPUT_SIZES[6] * sizeof(float));
    cudaMalloc(&d_c3_2, batchImg * OUTPUT_SIZES[7] * sizeof(float));
    cudaMalloc(&d_c3_3, batchImg * OUTPUT_SIZES[8] * sizeof(float));
    cudaMalloc(&d_p3,   batchImg * OUTPUT_SIZES[9] * sizeof(float));
    cudaMalloc(&d_c4_1, batchImg * OUTPUT_SIZES[10] * sizeof(float));
    cudaMalloc(&d_c4_2, batchImg * OUTPUT_SIZES[11] * sizeof(float));
    cudaMalloc(&d_c4_3, batchImg * OUTPUT_SIZES[12] * sizeof(float));
    cudaMalloc(&d_p4,   batchImg * OUTPUT_SIZES[13] * sizeof(float));
    cudaMalloc(&d_c5_1, batchImg * OUTPUT_SIZES[14] * sizeof(float));
    cudaMalloc(&d_c5_2, batchImg * OUTPUT_SIZES[15] * sizeof(float));
    cudaMalloc(&d_c5_3, batchImg * OUTPUT_SIZES[16] * sizeof(float));
    cudaMalloc(&d_p5,   batchImg * OUTPUT_SIZES[17] * sizeof(float));
    cudaMalloc(&d_fc1,  batchImg * OUTPUT_SIZES[18] * sizeof(float));
    cudaMalloc(&d_fc2,  batchImg * OUTPUT_SIZES[19] * sizeof(float));
    cudaMalloc(&d_fc3,  batchImg * OUTPUT_SIZES[20] * sizeof(float));*/
    show_mem_gpu("After malloc output vectors");
    

    // run network
    size_t image_size = batchImg*3*32*32 * sizeof(float);
    float *d_image;
    cudaMalloc(&d_image, image_size);

    int start_num_images = num_images%batchImg;
    // Images will processed by batch
    for(int i = start_num_images; i < num_images; i += batchImg2)
    {
        printf("i = %d\n", i);
        batchImg = batch_size;
        // Copy image from host to device
        float *image = images + i * 3 * 32 * 32;
        
        cudaEventRecord(start);
        cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        data_transfer_time += milliseconds/1000;

        convolution_layer_v2(d_image, d1, d_w1_1, d_b1_1, 64, 3, 32, batchImg);
        convolution_layer_v2(d1, d2, d_w1_2, d_b1_2, 64, 64, 32, batchImg);
        pooling_layer(d2, p1, 64, 16, batchImg);

        /////////////////
        // Copy image from host to device
        image = images + (i+batchImg) * 3 * 32 * 32;
        
        cudaEventRecord(start);
        cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        data_transfer_time += milliseconds/1000;

        convolution_layer_v2(d_image, d1, d_w1_1, d_b1_1, 64, 3, 32, batchImg);
        convolution_layer_v2(d1, d2, d_w1_2, d_b1_2, 64, 64, 32, batchImg);
        pooling_layer(d2, d1, 64, 16, batchImg);

        // copy p1 to d1
        batchImg = batchImg2;
        ////////////////

        convolution_layer_v2(d2, d1, d_w2_1, d_b2_1, 128, 64, 16, batchImg);
        convolution_layer_v2(d1, d2, d_w2_2, d_b2_2, 128, 128, 16, batchImg);
        pooling_layer(d2, d1, 128, 8, batchImg);

        convolution_layer_v2(d1, d2, d_w3_1, d_b3_1, 256, 128, 8, batchImg);
        convolution_layer_v2(d2, d1, d_w3_2, d_b3_2, 256, 256, 8, batchImg);
        convolution_layer_v2(d1, d2, d_w3_3, d_b3_3, 256, 256, 8, batchImg);
        pooling_layer(d2, d1, 256, 4, batchImg);

        convolution_layer_v2(d1, d2, d_w4_1, d_b4_1, 512, 256, 4, batchImg);
        convolution_layer_v2(d2, d1, d_w4_2, d_b4_2, 512, 512, 4, batchImg);
        convolution_layer_v2(d1, d2, d_w4_3, d_b4_3, 512, 512, 4, batchImg);
        pooling_layer(d2, d1, 512, 2, batchImg);

        convolution_layer_v2(d1, d2, d_w5_1, d_b5_1, 512, 512, 2, batchImg);
        convolution_layer_v2(d2, d1, d_w5_2, d_b5_2, 512, 512, 2, batchImg);
        convolution_layer_v2(d1, d2, d_w5_3, d_b5_3, 512, 512, 2, batchImg);
        pooling_layer(d2, d1, 512, 1, batchImg);

        fc_layer(d1, d2, d_w1, d_b1, 512, 512, batchImg);
        fc_layer(d2, d1, d_w2, d_b2, 512, 512, batchImg);
        fc_layer(d1, d2, d_w3, d_b3, 10, 512, batchImg);

        // Copy result from device memory to host memory
        float *fc3_mul  = alloc_layer(OUTPUT_SIZES[20] * batchImg);
        cudaEventRecord(start);
        cudaMemcpy(fc3_mul, d2, batchImg * OUTPUT_SIZES[20] * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        data_transfer_time += milliseconds/1000;

        // Predicted labels
        for (int j = 0; j < batchImg; j++) {
            float *fc3 = fc3_mul + j*10;
            softmax(fc3, 10);
            int idx = i + j;
            labels[idx] = find_max(fc3, 10);
            confidences[idx] = fc3[labels[idx]];
        }
        free(fc3_mul);
    }
    

    /*// The remaining images
    size_t image_size2 = start_num_images*3*32*32 * sizeof(float);
    float *d_image2;
    batchImg = start_num_images;
    for(int i = 0; i < start_num_images; i += start_num_images)
    {
        // Copy image from host to device
        float *image = images + i * 3 * 32 * 32;
        cudaEventRecord(start);
        cudaMalloc(&d_image2, image_size2);
        cudaMemcpy(d_image2, image, image_size2, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        data_transfer_time += milliseconds/1000;

        convolution_layer_v2(d_image2, d1, d_w1_1, d_b1_1, 64, 3, 32, batchImg);
        convolution_layer_v2(d1, d2, d_w1_2, d_b1_2, 64, 64, 32, batchImg);
        pooling_layer(d2, d1, 64, 16, batchImg);

        convolution_layer_v2(d2, d1, d_w2_1, d_b2_1, 128, 64, 16, batchImg);
        convolution_layer_v2(d1, d2, d_w2_2, d_b2_2, 128, 128, 16, batchImg);
        pooling_layer(d2, d1, 128, 8, batchImg);

        convolution_layer_v2(d1, d2, d_w3_1, d_b3_1, 256, 128, 8, batchImg);
        convolution_layer_v2(d2, d1, d_w3_2, d_b3_2, 256, 256, 8, batchImg);
        convolution_layer_v2(d1, d2, d_w3_3, d_b3_3, 256, 256, 8, batchImg);
        pooling_layer(d2, d1, 256, 4, batchImg);

        convolution_layer_v2(d1, d2, d_w4_1, d_b4_1, 512, 256, 4, batchImg);
        convolution_layer_v2(d2, d1, d_w4_2, d_b4_2, 512, 512, 4, batchImg);
        convolution_layer_v2(d1, d2, d_w4_3, d_b4_3, 512, 512, 4, batchImg);
        pooling_layer(d2, d1, 512, 2, batchImg);

        convolution_layer_v2(d1, d2, d_w5_1, d_b5_1, 512, 512, 2, batchImg);
        convolution_layer_v2(d2, d1, d_w5_2, d_b5_2, 512, 512, 2, batchImg);
        convolution_layer_v2(d1, d2, d_w5_3, d_b5_3, 512, 512, 2, batchImg);
        pooling_layer(d2, d1, 512, 1, batchImg);

        fc_layer(d1, d2, d_w1, d_b1, 512, 512, batchImg);
        fc_layer(d2, d1, d_w2, d_b2, 512, 512, batchImg);
        fc_layer(d1, d2, d_w3, d_b3, 10, 512, batchImg);

        // Copy result from device memory to host memory
        float *fc3_mul  = alloc_layer(OUTPUT_SIZES[20] * batchImg);
        cudaEventRecord(start);
        cudaMemcpy(fc3_mul, d2, batchImg * OUTPUT_SIZES[20] * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        data_transfer_time += milliseconds/1000;

        // Predicted labels
        for (int j = 0; j < batchImg; j++) {
            float *fc3 = fc3_mul + j*10;
            softmax(fc3, 10);
            int idx = i + j;
            labels[idx] = find_max(fc3, 10);
            confidences[idx] = fc3[labels[idx]];
        }
        free(fc3_mul);
    }
    cudaFree(d_image2);*/
    printf("data transfer time = %f s\n", data_transfer_time);
    printf("pooing time = %f s\n", pooling_time);
    printf("convolution time = %f s\n", conv_time);
    printf("fully connected time = %f s\n", fc_time);
    printf("softmax time = %f s\n", softmax_time);

    cudaFree(d_w1_1); cudaFree(d_b1_1); cudaFree(d_w1_2); cudaFree(d_b1_2);
    cudaFree(d_w2_1); cudaFree(d_b2_2); cudaFree(d_w2_2); cudaFree(d_b2_2);
    cudaFree(d_w3_1); cudaFree(d_b3_1); cudaFree(d_w3_2); cudaFree(d_b3_2); cudaFree(d_w3_3); cudaFree(d_b3_3);
    cudaFree(d_w4_1); cudaFree(d_b4_1); cudaFree(d_w4_2); cudaFree(d_b4_2); cudaFree(d_w4_3); cudaFree(d_b4_3);
    cudaFree(d_w5_1); cudaFree(d_b5_1); cudaFree(d_w5_2); cudaFree(d_b5_2); cudaFree(d_w5_3); cudaFree(d_b5_3);
    cudaFree(d_w1);   cudaFree(d_b1);   cudaFree(d_w2);   cudaFree(d_b2);   cudaFree(d_w3);   cudaFree(d_b3);

    /*cudaFree(d_c1_1); cudaFree(d_c1_2); cudaFree(d_p1);
    cudaFree(d_c2_1); cudaFree(d_c2_2); cudaFree(d_p2);
    cudaFree(d_c3_1); cudaFree(d_c3_2); cudaFree(d_c3_3); cudaFree(d_p3);
    cudaFree(d_c4_1); cudaFree(d_c4_2); cudaFree(d_c4_3); cudaFree(d_p4);
    cudaFree(d_c5_1); cudaFree(d_c5_2); cudaFree(d_c5_3); cudaFree(d_p5);
    cudaFree(d_fc1); cudaFree(d_fc2); cudaFree(d_fc3);*/

}
