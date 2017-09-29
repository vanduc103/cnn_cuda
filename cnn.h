#ifndef _CNN_H
#define _CNN_H

void cnn_init();
void cnn(float *images, float **network, int *labels, float *confidences, int num_images);
void cnn(float *images, float **network, int *labels, float *confidences, int num_images, int batch_size);

#endif 
