"""
Aim: To build and train an Artificial Neural Network (ANN) using TensorFlow and Keras for handwritten digit classification on
the MNIST dataset, and to evaluate it using accuracy, loss and training-validation curves.

Algorithm:

1. Import the required TensorFlow, Keras and Matplotlib libraries.
2. Load the MNIST dataset and normalize the pixel values to [0, 1].
3. Build a Sequential ANN with a Flatten layer, one hidden Dense layer (64, ReLU) and an output Dense layer (10, softmax).
4. Compile with the Adam optimizer and sparse categorical cross-entropy loss.
5. Train for 10 epochs with batch size 32 and 20% validation data.
6. Evaluate the model on the test data to get the loss and accuracy.
7. Plot the training and validation loss curves.
8. Plot the training and validation accuracy curves and analyze the performance.

Result:

Thus, the ANN model was successfully trained on the MNIST dataset and achieved high accuracy in handwritten digit classification.
The loss and accuracy curves showed effective learning and good model performance.
"""

import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
tf.keras.utils.set_random_seed(42)

(xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
xtr, xte = xtr / 255.0, xte / 255.0

model = Sequential([
    Input((28, 28)), Flatten(),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
h = model.fit(xtr, ytr, epochs=10, batch_size=32, validation_split=0.2)
loss, acc = model.evaluate(xte, yte, verbose=0)
print("Test Loss:", loss, "| Test Accuracy:", acc)

plt.figure(figsize=(12, 5))
for i, k in enumerate(["loss", "accuracy"], 1):
    plt.subplot(1, 2, i); plt.plot(h.history[k], label="Training"); plt.plot(h.history["val_" + k], label="Validation")
    plt.title(k.title() + " Curve"); plt.xlabel("Epoch"); plt.legend()
plt.show()