

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

## Future Work

- Add support for multiple Conv2D layers.  
- Implement UART output for digit prediction.  
- Extend to full MNIST test set evaluation.  
```

Would you like me to also add a **diagram** (ASCII or markdown table) showing the CNN architecture flow: `Input → Conv2D → MaxPool → Flatten → Dense → Output`? That would make the README more visually intuitive.
