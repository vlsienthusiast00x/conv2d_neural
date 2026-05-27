

# Conv2D MNIST Inference on VSDSquadron PRO

This repository demonstrates how to implement a **Conv2D neural network layer** on the VSDSquadron PRO RISC-V board.  
The workflow includes training a CNN in Python using TensorFlow/Keras, exporting quantized weights and biases to C header files, and running inference in C on the embedded hardware.

---

## Project Overview

- **Python (TensorFlow/Keras)**  
  - Train a simple CNN on MNIST digits.  
  - Quantize Conv2D and Dense layer weights to `int8`.  
  - Export weights, biases, and scale factors to `weights.h`.  
  - Export one MNIST sample to `input.h`.

- **C (VSDSquadron PRO)**  
  - Implement Conv2D with ReLU + MaxPooling.  
  - Implement Dense layer with quantized weights.  
  - Perform inference on the exported MNIST sample.  
  - Print the predicted digit via UART/terminal.

---

## Files

### Python Script
- Defines CNN: `Conv2D → MaxPooling → Flatten → Dense`.
- Trains briefly on MNIST dataset.
- Quantizes weights and biases.
- Exports:
  - `weights.h` → Conv2D and Dense layer parameters.
  - `input.h` → One MNIST sample input.

### C Program (`main.c`)
- Implements:
  - `conv_relu_maxpool_2d_gray()` → Conv2D + ReLU + MaxPooling.
  - `dense_layer()` → Fully connected layer with quantized weights.
  - `argmax()` → Selects predicted digit.
- Runs inference on the sample input.
- Prints predicted digit.

---

## Workflow

1. **Train and Export Weights (Python)**
   ```bash
   python export_weights.py
   ```
   Generates:
   - `weights.h`
   - `input.h`

2. **Build and Flash (C on VSDSquadron PRO)**
   - Include generated headers in your project.
   - Compile `main.c` with Freedom Studio toolchain.
   - Flash to VSDSquadron PRO.

3. **Run Inference**
   - Board executes CNN inference.
   - Predicted digit is printed to terminal.

---

## Example Output

```text
Predicted digit: 5
```

---

## Project Structure

```
├── export_weights.py   # Python script for training + export
├── weights.h           # Quantized Conv2D + Dense weights/biases/scales
├── input.h             # MNIST sample input
├── main.c              # C implementation of Conv2D + Dense inference
└── README.md           # Project documentation
```

---

## Notes

- Quantization:  
  - Weights → `int8`  
  - Biases → `float32`  
  - Scale factors → `float`  

- Tested on:  
  - **VSDSquadron PRO board**  
  - **Freedom Studio toolchain**

---

# Breakdown of the main program

## C Inference Code Explanation (`main.c`)

This file runs the exported MNIST Conv2D model on the VSDSquadron PRO board. It uses quantized weights and biases from `weights.h` and a sample input from `input.h`.

### Key Definitions
```c
#define INP_DIM        28   // Input image size (28x28)
#define OUT_CH         8    // Number of Conv2D filters
#define KERNEL_H       3    // Kernel height
#define KERNEL_W       3    // Kernel width
#define STRIDE         1    // Convolution stride
#define OUT_DIM        ((INP_DIM - KERNEL_H) / STRIDE + 1) // Conv output size = 26
#define FC_OUT         10   // Dense layer outputs (digits 0–9)
#define POOL_DIM       (OUT_DIM / 2)  // After 2x2 pooling → 13
#define MAXPOOL_OUT    (OUT_CH * POOL_DIM * POOL_DIM) // Flattened size = 1352
```

These constants define the CNN architecture dimensions for inference.

---

### Conv2D + ReLU + MaxPooling
```c
void conv_relu_maxpool_2d_gray(const float *input, float *output,
                               const int8_t *weights, const float *biases, float scale,
                               int inp_dim, int out_ch,
                               int kh, int kw, int stride, int pool_stride)
```
- Performs convolution with quantized weights.  
- Adds bias and rescales using the quantization scale.  
- Applies **ReLU activation** (negative values → 0).  
- Applies **2×2 max pooling** to reduce spatial dimensions.  
- Stores results in `maxpool_out`.

---

### Dense Layer
```c
void dense_layer(const int8_t *weights, const float *biases, float scale,
                 const float *input, float *output,
                 int in_size, int out_size, int apply_relu)
```
- Fully connected layer implementation.  
- Multiplies input vector with quantized weights.  
- Adds bias and rescales.  
- Optionally applies ReLU.  
- Stores results in `dense_out`.

---

### Argmax
```c
int argmax(const float *logits, int size)
```
- Finds the index of the maximum value in the output vector.  
- This index corresponds to the predicted digit (0–9).

---

### Prediction Pipeline
```c
int predict(const float *input) {
    conv_relu_maxpool_2d_gray(input, maxpool_out,
                              conv1_weight, conv1_bias, conv1_scale,
                              INP_DIM, OUT_CH,
                              KERNEL_H, KERNEL_W, STRIDE, 2);

    dense_layer(fc1_weight, fc1_bias, fc1_scale,
                maxpool_out, dense_out, MAXPOOL_OUT, FC_OUT, 0);

    return argmax(dense_out, FC_OUT);
}
```
- Runs the Conv2D → MaxPool → Dense pipeline.  
- Returns the predicted digit.

---

### Main Function
```c
int main(void) {
    int predicted_digit = predict(input);
    printf("Predicted digit: %d\n", predicted_digit);
    return 0;
}
```
- Calls `predict()` with the MNIST sample from `input.h`.  
- Prints the predicted digit to the terminal.

---

### Summary
- **Conv2D layer** extracts features from the 28×28 input image.  
- **MaxPooling** reduces dimensionality.  
- **Dense layer** maps features to digit classes (0–9).  
- **Argmax** selects the most likely digit.  
- The final output is printed as:

```text
Predicted digit: 5
```

---

