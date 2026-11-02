"""
Aim: To build an image captioning model that generates a sentence for an image, using InceptionV3 image features and an LSTM
decoder trained on the Flickr8k dataset.

Algorithm:

1. Load the train, validation and test image names and the captions of the Flickr8k dataset.
2. Clean each caption (lowercase, remove punctuation) and add the words startseq and endseq.
3. Extract a 2048-value feature vector for every image using the pre-trained InceptionV3 model.
4. Build a tokenizer from the training captions and find the vocabulary size and the maximum caption length.
5. Create training data: for each caption, the image features and the first i words are the input and the next word is the output.
6. Build the model: image features go through Dropout and Dense (256), the caption goes through Embedding, Dropout and LSTM (256).
   Add both, then Dense (256) and a Dense softmax output.
7. Compile with the Adam optimizer and sparse categorical cross-entropy loss, then train for 5 epochs with batch size 256.
8. To caption an image, start with startseq, predict one word at a time, and stop at endseq or the maximum length.

Result:

Thus, an image captioning model was successfully developed and trained using the Flickr8k dataset. The trained model generated
a caption for the given test image.
"""

import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, add
tf.keras.utils.set_random_seed(42)
import os, string
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

img_dir, txt_dir = r"Flicker8k_Dataset", r"Flickr8k_text"
names = lambda f: [n for n in (l.strip() for l in open(f"{txt_dir}/{f}", encoding="utf-8")) if os.path.isfile(f"{img_dir}/{n}")]
train_n, val_n, test_n = names("Flickr_8k.trainImages.txt"), names("Flickr_8k.devImages.txt"), names("Flickr_8k.testImages.txt")
print("Train:", len(train_n), "| Val:", len(val_n), "| Test:", len(test_n))

caps = {}
for l in open(f"{txt_dir}/Flickr8k.token.txt", encoding="utf-8"):
    p = l.strip().split("\t")
    if len(p) == 2:
        c = p[1].lower().translate(str.maketrans("", "", string.punctuation))
        caps.setdefault(p[0].split("#")[0], []).append(f"startseq {c} endseq")

inception = InceptionV3(weights="imagenet", pooling="avg")
feat = lambda n: inception.predict(preprocess_input(img_to_array(load_img(f"{img_dir}/{n}", target_size=(299, 299)))[None]), verbose=0)[0]
feats = {n: feat(n) for n in train_n + val_n + test_n}

train_caps = [c for n in train_n for c in caps[n]]
tok = Tokenizer(); tok.fit_on_texts(train_caps)
V, L = len(tok.word_index) + 1, max(len(c.split()) for c in train_caps)
print("Vocabulary size:", V, "| Max caption length:", L)

def seqs(ns):
    X1, X2, y = [], [], []
    for n in ns:
        for c in caps[n]:
            s = tok.texts_to_sequences([c])[0]
            for i in range(1, len(s)):
                X1.append(feats[n]); X2.append(s[:i]); y.append(s[i])
    return np.array(X1), pad_sequences(X2, maxlen=L), np.array(y)
X1, X2, y = seqs(train_n)
X1v, X2v, yv = seqs(val_n)

i1, i2 = Input((2048,)), Input((L,))
a = Dense(256, activation="relu")(Dropout(0.3)(i1))
b = LSTM(256)(Dropout(0.3)(Embedding(V, 256, mask_zero=True)(i2)))
model = Model([i1, i2], Dense(V, activation="softmax")(Dense(256, activation="relu")(add([a, b]))))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
model.summary()
h = model.fit([X1, X2], y, validation_data=([X1v, X2v], yv), epochs=5, batch_size=256)
plt.plot(h.history["loss"], label="Training Loss"); plt.plot(h.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Image Captioning Model Loss"); plt.legend(); plt.show()

def caption(n):
    t = "startseq"
    for _ in range(L):
        p = model.predict([feats[n][None], pad_sequences(tok.texts_to_sequences([t]), maxlen=L)], verbose=0).argmax()
        w = tok.index_word.get(p)
        if w in (None, "endseq"): break
        t += " " + w
    return t.replace("startseq", "").strip()

n = test_n[0]
plt.imshow(load_img(f"{img_dir}/{n}")); plt.title(caption(n)); plt.axis("off"); plt.show()