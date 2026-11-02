"""
Aim: To develop a Convolutional Neural Network (CNN) model using TensorFlow to recognize and classify different persons from
facial images in the LFW (Labeled Faces in the Wild) dataset.

Algorithm:

1. Import NumPy, TensorFlow, Matplotlib and Scikit-learn libraries.
2. Load the LFW face dataset with a minimum of 70 images per person.
3. Extract the face images and the target labels.
4. Normalize the pixel values and add the channel dimension.
5. Split the data into 80% training and 20% testing using stratified sampling.
6. Build a CNN using Conv2D, ReLU, MaxPooling, Flatten and Dense layers.
7. Compile with the Adam optimizer and sparse categorical cross-entropy loss, then train for 10 epochs.
8. Evaluate on the test data and predict the identity of a sample test face.

Result:

Thus, a CNN-based face recognition model was successfully implemented using the LFW dataset. The trained model was evaluated on
the test data and predicted the identity of a sample facial image.
"""

import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
tf.keras.utils.set_random_seed(42)
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.5)
X, y, names = lfw.images[..., None] / 255.0, lfw.target, lfw.target_names
for n, c in zip(names, np.bincount(y)): print(n, ":", c, "images")
print("Total images:", len(X), "| Total people:", len(names))

xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Sequential([
    Input(X.shape[1:]),
    Conv2D(32, 3, activation="relu"), MaxPooling2D(),
    Conv2D(64, 3, activation="relu"), MaxPooling2D(),
    Flatten(), Dense(128, activation="relu"),
    Dense(len(names), activation="softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
h = model.fit(xtr, ytr, epochs=10, batch_size=32, validation_split=0.2)
loss, acc = model.evaluate(xte, yte, verbose=0)
print("Test Loss:", loss, "| Test Accuracy:", acc)

p = model.predict(xte[:1], verbose=0).argmax()
print("Predicted person:", names[p], "\nActual person   :", names[yte[0]])
plt.imshow(xte[0].squeeze(), cmap="gray"); plt.title("Predicted: " + names[p]); plt.axis("off"); plt.show()