"""
Aim: To build an LSTM model that classifies IMDb movie reviews as Positive or Negative, and to test it on a custom sentence.

Algorithm:

1. Import TensorFlow and load the IMDb data, keeping only the top 10,000 most common words.
2. Pad every review to a fixed length of 200 words.
3. Build the model with an Embedding layer (10,000 × 128), an LSTM layer (64 units) and a Dense layer (1 unit, sigmoid).
4. Compile with the Adam optimizer, binary cross-entropy loss and accuracy as the metric.
5. Train for 3 epochs with batch size 64.
6. Evaluate on the test data and print the test accuracy.
7. Load the IMDb word index and convert the sentence "This movie was super" into numbers, then pad it to 200.
8. Predict the score. If the score is 0.5 or more, the sentiment is Positive, otherwise Negative.

Result:

Thus, an LSTM-based sentiment analysis model was successfully developed and trained using the IMDb movie review data. The trained model 
successfully classified movie reviews into positive or negative sentiments.
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
tf.keras.utils.set_random_seed(42)

(xtr, ytr), (xte, yte) = imdb.load_data(num_words=10000)
xtr, xte = pad_sequences(xtr, maxlen=200), pad_sequences(xte, maxlen=200)

model = Sequential([
    Embedding(10000, 128),
    LSTM(64),
    Dense(1, activation="sigmoid")])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(xtr, ytr, epochs=3, batch_size=64)
print("Test Accuracy:", model.evaluate(xte, yte)[1])

wi = imdb.get_word_index()
def encode(sentence):
    ids = [1]
    for w in sentence.lower().split():
        i = wi.get(w.strip(".,!?"), None)
        ids.append(i + 3 if i is not None and i + 3 < 10000 else 2)
    return pad_sequences([ids], maxlen=200)

sentence = "This movie was super"
p = model.predict(encode(sentence))[0][0]
print("Sentence:", sentence, "\nSentiment:", "Positive" if p >= 0.5 else "Negative", "\nScore:", p)