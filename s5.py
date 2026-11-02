"""
Aim: To build a character-level language model using a SimpleRNN in TensorFlow. The model learns from the Shakespeare text 
and generates new text from a starting phrase.

Algorithm:

1. Import TensorFlow and NumPy, then load the Shakespeare text file.
2. Find all unique characters and give each one a number (vocabulary).
3. Convert the whole text into numbers using this mapping.
4. Cut the text into sequences of 100 characters. The target is the same sequence shifted by one character.
5. Shuffle the sequences and group them into batches of 64.
6. Build the model with an Embedding layer (128), a SimpleRNN layer (256 units) and a Dense output layer (65 units).
7. Compile with the Adam optimizer and sparse categorical cross-entropy loss (with logits), then train for 10 epochs.
8. Start with "Shall ", predict one character at a time using the last 100 characters, and add each prediction to the text.

Result:

Thus, a character-level language model using SimpleRNN was successfully developed and trained on the given text corpus. The trained model
successfully generated new text based on the learned character patterns.
"""

import numpy as np, tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
tf.keras.utils.set_random_seed(42)

text = open(r"shakespeare.txt", encoding="utf-8").read()
vocab = sorted(set(text)); V = len(vocab)
c2i = {c: i for i, c in enumerate(vocab)}
ids = np.array([c2i[c] for c in text], dtype="int32")
print("Corpus length:", len(text), "| Vocab size:", V)

w = np.lib.stride_tricks.sliding_window_view(ids, 101)
ds = tf.data.Dataset.from_tensor_slices((w[:, :-1], w[:, 1:])).shuffle(10000).batch(64, drop_remainder=True)

model = Sequential([
    Embedding(V, 128),
    SimpleRNN(256, return_sequences=True),
    Dense(V)])
model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()
model.fit(ds, epochs=10)

def generate(start, n=200):
    out = start
    for _ in range(n):
        x = np.array([[c2i[c] for c in out[-100:]]])
        out += vocab[model.predict(x, verbose=0)[0, -1].argmax()]
    return out

print("\nGenerated Text:\n", generate("Shall ", 200))