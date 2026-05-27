import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import Model

# --- Define model ---
model = Sequential([
    Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2), strides=2),   # reduces 26x26 → 13x13
    Flatten(),
    Dense(10, activation='softmax')
])

# --- Load MNIST ---
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, -1)

# --- Train briefly ---
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=128)

# --- Conv2D layer ---
conv_layer = model.layers[0]
conv_w, conv_b = conv_layer.get_weights()
conv_scale = np.max(np.abs(conv_w)) / 127.0
conv_w_q = np.round(conv_w / conv_scale).astype(np.int8)
conv_b_f = conv_b.astype(np.float32)

# --- Dense layer (index 3 after MaxPool + Flatten) ---
dense_layer = model.layers[3]
dense_w, dense_b = dense_layer.get_weights()
dense_scale = np.max(np.abs(dense_w)) / 127.0
dense_w_q = np.round(dense_w / dense_scale).astype(np.int8)
dense_b_f = dense_b.astype(np.float32)
conv_w_flat = np.transpose(conv_w_q, (3, 0, 1, 2)).flatten()
dense_w_flat = dense_w_q.T.flatten()

# --- Compute dimensions dynamically ---
OUT_CH = conv_layer.filters
OUT_DIM = (28 - 3) + 1   # conv output dimension = 26
POOL_DIM = OUT_DIM // 2  # after 2x2 pooling → 13
MAXPOOL_OUT = OUT_CH * POOL_DIM * POOL_DIM  # 8 * 13 * 13 = 1352
FC_OUT = dense_layer.units  # 10

# --- Write to weights.h ---
with open("weights.h", "w") as f:
    f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")

    # Conv2D
    f.write("// Conv2D layer: 8 filters, 3x3 kernel, 1 input channel\n")
    f.write(f"static const int8_t conv1_weight[{conv_w_flat.size}] = {{\n")
    f.write(",".join(map(str, conv_w_flat)))
    f.write("};\n\n")
    f.write(f"static const float conv1_bias[{conv_b_f.size}] = {{")
    f.write(",".join([f"{x:.6f}" for x in conv_b_f]))
    f.write("};\n\n")
    f.write(f"static const float conv1_scale = {conv_scale:.6f};\n\n")

    # Dense
    f.write(f"// Dense layer: {MAXPOOL_OUT} inputs -> {FC_OUT} outputs\n")
    f.write(f"static const int8_t fc1_weight[{dense_w_flat.size}] = {{\n")
    f.write(",".join(map(str, dense_w_flat)))
    f.write("};\n\n")
    f.write(f"static const float fc1_bias[{dense_b_f.size}] = {{")
    f.write(",".join([f"{x:.6f}" for x in dense_b_f]))
    f.write("};\n\n")
    f.write(f"static const float fc1_scale = {dense_scale:.6f};\n\n")

    f.write("#endif // WEIGHTS_H\n")

print("Exported Conv2D and Dense weights/biases/scales to weights.h")

model(x_train[:1])
sub_model = Model(inputs=model.inputs, outputs=model.layers[2].output)  # layer 2 = Flatten
sample = x_train[0:1]
flattened = sub_model.predict(sample)
print(flattened.shape)  # should be (1, 1352)
print(flattened[0, :20])  # first 20 values

# --- Write one MNIST sample to input.h ---
sample = x_train[7:8]          # shape: (1, 28, 28, 1)
sample_label = y_train[7]
input_flat = sample[0, :, :, 0].astype(np.float32).flatten()

with open("input.h", "w") as f:
    f.write("#ifndef INPUT_H\n#define INPUT_H\n\n")
    f.write("#include <stdint.h>\n\n")

    f.write(f"// MNIST sample label: {sample_label}\n")
    f.write(f"static const float input[{input_flat.size}] = {{\n")

    for i, x in enumerate(input_flat):
        f.write(f"{x:.8f}")
        if i != input_flat.size - 1:
            f.write(",")
        if (i + 1) % 12 == 0:
            f.write("\n")

    f.write("\n};\n\n")
    f.write("#endif // INPUT_H\n")

print(f"Exported MNIST sample to input.h, label = {sample_label}")
