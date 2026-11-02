"""
Aim: To build and train a CNN using TensorFlow and Keras for handwritten character classification on the EMNIST Balanced dataset,
and to evaluate it using accuracy, loss curves and a sample prediction.

Algorithm:

1. Import NumPy, Matplotlib, TensorFlow and gzip, then read the EMNIST Balanced files.
2. Normalize the pixel values and reshape the images to 28 x 28 x 1.
3. Build a Sequential CNN with two Conv2D + MaxPooling blocks, Flatten, Dense (128) and Dropout (0.5), and a Dense output layer (47).
4. Compile with the Adam optimizer and sparse categorical cross-entropy loss.
5. Train for 4 epochs with batch size 64 and 20% validation data.
6. Evaluate the model on the test data to get the loss and accuracy.
7. Plot the training and validation accuracy and loss curves.
8. Predict one test image, convert the label to a character using the mapping file, and show the result with its confidence.

Result:

Thus, the CNN model was successfully trained on the EMNIST Balanced dataset and achieved good accuracy in classifying handwritten
characters across 47 classes. The curves confirmed effective learning.
"""

import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
tf.keras.utils.set_random_seed(42)
import gzip

def idx(split, kind, off):
    return np.frombuffer(gzip.open(f"emnist-balanced-{split}-{kind}-ubyte.gz").read(), np.uint8, offset=off)

xtr, ytr = idx("train", "images-idx3", 16).reshape(-1, 28, 28, 1) / 255.0, idx("train", "labels-idx1", 8)
xte, yte = idx("test", "images-idx3", 16).reshape(-1, 28, 28, 1) / 255.0, idx("test", "labels-idx1", 8)
print("Train:", xtr.shape, "| Test:", xte.shape, "| Classes:", len(np.unique(ytr)))

model = Sequential([
    Input((28, 28, 1)),
    Conv2D(32, 3, activation="relu"), MaxPooling2D(),
    Conv2D(64, 3, activation="relu"), MaxPooling2D(),
    Flatten(), Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(47, activation="softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
h = model.fit(xtr, ytr, epochs=4, batch_size=64, validation_split=0.2)
loss, acc = model.evaluate(xte, yte, verbose=0)
print("Test Loss:", loss, "| Test Accuracy:", acc)

plt.figure(figsize=(12, 5))
for i, k in enumerate(["loss", "accuracy"], 1):
    plt.subplot(1, 2, i); plt.plot(h.history[k], label="Training"); plt.plot(h.history["val_" + k], label="Validation")
    plt.title(k.title() + " Curve"); plt.xlabel("Epoch"); plt.legend()
plt.show()
mp = {int(a): chr(int(b)) for a, b in (l.split() for l in open("emnist-balanced-mapping.txt"))}
i = 25
p = model.predict(xte[i:i + 1], verbose=0)[0]
plt.imshow(xte[i].squeeze(), cmap="gray"); plt.axis("off"); plt.show()
print("Actual    :", mp[yte[i]], "\nPredicted :", mp[p.argmax()], "\nConfidence:", round(p.max() * 100, 2), "%")