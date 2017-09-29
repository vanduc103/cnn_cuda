#include <stdio.h>
#include <stdlib.h>
#include "cnn.h"
#include "timer.h"

const char *CLASS_NAME[] = {
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
};

void print_usage_and_exit(char **argv) {
    fprintf(stderr, "Usage: %s <number of image> <output>\n", argv[0]);
    fprintf(stderr, " e.g., %s 3000 result.out\n", argv[0]);
    exit(EXIT_FAILURE);
}

void* read_bytes(const char *fn, size_t n) {
    FILE *f = fopen(fn, "rb");
    void *bytes = malloc(n);
    size_t r = fread(bytes, 1, n, f);
    fclose(f);
    if (r != n) {
        fprintf(stderr,
                "%s: %zd bytes are expected, but %zd bytes are read.\n",
                fn, n, r);
        exit(EXIT_FAILURE);
    }
    return bytes;
}

/*
 * Read images from "cifar10_image.bin".
 * CIFAR-10 test dataset consists of 10000 images with (3, 32, 32) size.
 * Thus, 10000 * 3 * 32 * 32 * sizeof(float) = 122880000 bytes are expected.
 */
const int IMAGE_CHW = 3 * 32 * 32 * sizeof(float);
float* read_images(size_t n) {
    return (float*)read_bytes("cifar10_image.bin", n * IMAGE_CHW);
}

/*
 * Read labels from "cifar10_label.bin".
 * 10000 * sizeof(int) = 40000 bytes are expected.
 */
int* read_labels(size_t n) {
    return (int*)read_bytes("cifar10_label.bin", n * sizeof(int));
}

/*
 * Read network from "network.bin".
 * conv1_1 : weight ( 64,   3, 3, 3) bias ( 64)
 * conv1_2 : weight ( 64,  64, 3, 3) bias ( 64)
 * conv2_1 : weight (128,  64, 3, 3) bias (128)
 * conv2_2 : weight (128, 128, 3, 3) bias (128)
 * conv3_1 : weight (256, 128, 3, 3) bias (256)
 * conv3_2 : weight (256, 256, 3, 3) bias (256)
 * conv3_3 : weight (256, 256, 3, 3) bias (256)
 * conv4_1 : weight (512, 256, 3, 3) bias (512)
 * conv4_2 : weight (512, 512, 3, 3) bias (512)
 * conv4_3 : weight (512, 512, 3, 3) bias (512)
 * conv5_1 : weight (512, 512, 3, 3) bias (512)
 * conv5_2 : weight (512, 512, 3, 3) bias (512)
 * conv5_3 : weight (512, 512, 3, 3) bias (512)
 * fc1     : weight (512, 512) bias (512)
 * fc2     : weight (512, 512) bias (512)
 * fc3     : weight ( 10, 512) bias ( 10)
 * Thus, 60980520 bytes are expected.
 */
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

float* read_network() {
    return (float*)read_bytes("network.bin", 60980520);
}

float** slice_network(float *p) {
    float **r = (float**)malloc(sizeof(float*) * 32); 
    for (int i = 0; i < 32; ++i) {
        r[i] = p;
        p += NETWORK_SIZES[i];
    }
    return r;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        print_usage_and_exit(argv);
    }

    int num_images = atoi(argv[1]);
    int batch_size = atoi(argv[3]);
    float *images = read_images(num_images);
    float *network = read_network();
    float **network_sliced = slice_network(network);
    int *labels = (int*)calloc(num_images, sizeof(int));
    float *confidences = (float*)calloc(num_images, sizeof(float));

    cnn_init();
    timer_start(0);
    cnn(images, network_sliced, labels, confidences, num_images, batch_size);
    printf("Elapsed time: %f sec\n", timer_end(0));

    FILE *of = fopen(argv[2], "w");
    int *labels_ans = read_labels(num_images);
    double acc = 0;
    for (int i = 0; i < num_images; ++i) {
        fprintf(of, "Image %04d: %s %f\n", i, CLASS_NAME[labels[i]], confidences[i]);
        if (labels[i] == labels_ans[i]) ++acc;
    }
    fprintf(of, "Accuracy: %f\n", acc / num_images);
    fclose(of);

    free(images);
    free(network);
    free(network_sliced);
    free(labels);
    free(confidences);
    free(labels_ans);

    return 0;
}
