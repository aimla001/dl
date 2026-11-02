"""
Aim: To build a GAN that generates synthetic Fashion-MNIST images, and to check whether adding these images to the training data improves classifier accuracy.

Algorithm:

1. Load the Fashion-MNIST dataset. Scale the images to [-1, 1] for the GAN and to [0, 1] for the classifier.
2. Build the Generator, which turns a 100-value noise vector into a 28×28 image.
3. Build the Discriminator, which uses Conv2D, LeakyReLU, Dropout and a sigmoid output to decide if an image is real or fake.
4. Join the Generator and Discriminator into a GAN model, keeping the Discriminator frozen inside it.
5. Train for 50 epochs with batch size 128. In each step, train the Discriminator on real images (label 0.9) and fake images (label 0), 
   then train the Generator to make the Discriminator say "real".
6. Display 16 generated images, then generate 5,000 synthetic images.
7. Train a CNN classifier on the original data only (3 epochs), then train an identical CNN on the original data plus the 5,000 synthetic 
   images (labelled using the first classifier).
8. Test both classifiers on the test set, then compare their accuracies and show them in a bar chart.

Result:

Thus, a GAN-based synthetic data generation and data augmentation model was successfully developed and trained using the Fashion-MNIST 
dataset. The trained GAN successfully generated synthetic images, which were used to augment the training data for image classification.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, LeakyReLU, Reshape,
    Conv2DTranspose, Conv2D, Dropout, Flatten, MaxPooling2D
)

tf.keras.utils.set_random_seed(42)

(xr, ytr), (xt, yte) = tf.keras.datasets.fashion_mnist.load_data()
xr, xt = xr[..., None].astype("float32"), xt[..., None].astype("float32")
x_gan = (xr - 127.5) / 127.5                                    
x_cls, x_test = xr / 255.0, xt / 255.0                          

G = Sequential([Input((100,)),
    Dense(7 * 7 * 256, use_bias=False), BatchNormalization(), LeakyReLU(), Reshape((7, 7, 256)),
    Conv2DTranspose(128, 4, 2, padding="same", use_bias=False), BatchNormalization(), LeakyReLU(),
    Conv2DTranspose(64, 4, 2, padding="same", use_bias=False), BatchNormalization(), LeakyReLU(),
    Conv2D(1, 3, padding="same", activation="tanh")])

D = Sequential([Input((28, 28, 1)),            
    Conv2D(64, 4, 2, padding="same"), LeakyReLU(0.2), Dropout(0.3),
    Conv2D(128, 4, 2, padding="same"), LeakyReLU(0.2), Dropout(0.3),
    Flatten(), Dense(1, activation="sigmoid")])
D.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy")

D.trainable = False                                             
GAN = Sequential([G, D])
GAN.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy")

noise = lambda n: np.random.normal(0, 1, (n, 100)).astype("float32")
for epoch in range(50):
    for i in range(0, len(x_gan), 128):
        real = x_gan[i:i + 128]; n = len(real)
        fake = G(noise(n), training=True).numpy()               
        D.trainable = True
        D.train_on_batch(real, np.full((n, 1), 0.9))           
        D.train_on_batch(fake, np.zeros((n, 1)))
        D.trainable = False
        GAN.train_on_batch(noise(n), np.ones((n, 1)))           
    print("Epoch:", epoch + 1)

gen = (G.predict(noise(16), verbose=0) + 1) / 2                 
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1); plt.imshow(gen[i].squeeze(), cmap="gray"); plt.axis("off")
plt.suptitle("GAN Generated Images"); plt.tight_layout(); plt.show()

num_fake = 5000
fake_images = (G.predict(noise(num_fake), verbose=0) + 1) / 2
print("Generated images:", fake_images.shape)

def classifier():
    m = Sequential([Input((28, 28, 1)),
        Conv2D(32, 3, activation="relu"), MaxPooling2D(),
        Conv2D(64, 3, activation="relu"), MaxPooling2D(),
        Flatten(), Dense(64, activation="relu"), Dense(10, activation="softmax")])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

print("Training original classifier...")
m_orig = classifier(); m_orig.fit(x_cls, ytr, epochs=3, batch_size=128)
acc_orig = m_orig.evaluate(x_test, yte, verbose=0)[1]

fake_labels = m_orig.predict(fake_images, verbose=0).argmax(1)

x_aug = np.concatenate([x_cls, fake_images]); y_aug = np.concatenate([ytr, fake_labels])
idx = np.random.permutation(len(x_aug)); x_aug, y_aug = x_aug[idx], y_aug[idx]
print("Augmented data:", x_aug.shape)

print("Training augmented classifier...")
m_aug = classifier(); m_aug.fit(x_aug, y_aug, epochs=3, batch_size=128)
acc_aug = m_aug.evaluate(x_test, yte, verbose=0)[1]

print(f"\nOriginal Accuracy: {acc_orig*100:.2f} %\nGAN Augmented Accuracy: {acc_aug*100:.2f} %\nChange: {(acc_aug-acc_orig)*100:.2f} %")
plt.bar(["Original", "GAN Augmented"], [acc_orig * 100, acc_aug * 100])
plt.ylabel("Accuracy (%)"); plt.title("Classification Accuracy"); plt.ylim(0, 100); plt.show()