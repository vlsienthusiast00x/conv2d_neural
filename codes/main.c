// main.c
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "weights.h"
#include "input.h"


#define INP_DIM        28
#define OUT_CH         8
#define KERNEL_H       3
#define KERNEL_W       3
#define STRIDE         1
#define OUT_DIM        ((INP_DIM - KERNEL_H) / STRIDE + 1)
#define CONV_OUT_SIZE  (OUT_CH * OUT_DIM * OUT_DIM)
#define FC_OUT         10
#define POOL_DIM       (OUT_DIM / 2)
#define MAXPOOL_OUT    (OUT_CH * POOL_DIM * POOL_DIM)


static float maxpool_out[MAXPOOL_OUT];
static float dense_out[FC_OUT];

void conv_relu_maxpool_2d_gray(const float *input, float *output,
                               const int8_t *weights, const float *biases, float scale,
                               int inp_dim, int out_ch,
                               int kh, int kw, int stride, int pool_stride) {
    int conv_dim = (inp_dim - kh) / stride + 1;
    int pool_dim = conv_dim / pool_stride;

    for (int c = 0; c < out_ch; c++) {
        int w_base = c * kh * kw;

        for (int pr = 0; pr < pool_dim; pr++) {
            for (int pc = 0; pc < pool_dim; pc++) {
                float maxval = -3.402823e38f;

                for (int rr = 0; rr < pool_stride; rr++) {
                    for (int cc = 0; cc < pool_stride; cc++) {
                        int conv_r = pr * pool_stride + rr;
                        int conv_c = pc * pool_stride + cc;

                        float acc = biases[c];
                        int32_t sum = 0;

                        for (int fr = 0; fr < kh; fr++) {
                            for (int fc = 0; fc < kw; fc++) {
                                int inp_index = (conv_r + fr) * inp_dim + (conv_c + fc);
                                int w_index = w_base + fr * kw + fc;

                                sum += (int32_t)(input[inp_index] * 127.0f) * weights[w_index];
                            }
                        }

                        acc += ((float)sum * scale) / 127.0f;

                        if (acc < 0.0f) {
                            acc = 0.0f;
                        }

                        if (acc > maxval) {
                            maxval = acc;
                        }
                    }
                }

                output[(pr * pool_dim + pc) * out_ch + c] = maxval;
            }
        }
    }
}

// Dense layer: weights layout [out_features, in_features]
void dense_layer(const int8_t *weights, const float *biases, float scale,
                 const float *input, float *output,
                 int in_size, int out_size, int apply_relu) {
    for (int o = 0; o < out_size; o++) {
        float acc = biases[o];
        int32_t sum = 0;
        int w_base = o * in_size;
        for (int i = 0; i < in_size; i++) {
            sum += (int32_t)(input[i] * 127.0f) * weights[w_base + i];
        }
        acc += ((float)sum * scale) / 127.0f;
        output[o] = apply_relu ? (acc > 0.0f ? acc : 0.0f) : acc;
    }
}

int argmax(const float *logits, int size) {
    int max_index = 0;
    float max_val = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_index = i;
        }
    }
    return max_index;
}

int predict(const float *input) {
    conv_relu_maxpool_2d_gray(input, maxpool_out,
                              conv1_weight, conv1_bias, conv1_scale,
                              INP_DIM, OUT_CH,
                              KERNEL_H, KERNEL_W, STRIDE, 2);

    dense_layer(fc1_weight, fc1_bias, fc1_scale,
                maxpool_out, dense_out, MAXPOOL_OUT, FC_OUT, 0);

    return argmax(dense_out, FC_OUT);
}

int main(void) {

    int predicted_digit = predict(input);
    printf("Predicted digit: %d\n", predicted_digit);

    return 0;
}
