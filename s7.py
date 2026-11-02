

"""
Aim: To build an Encoder–Decoder LSTM model that gives each word in a sentence its POS tag, using the Penn Treebank corpus.

Algorithm:

1. Load the tagged sentences from the Penn Treebank and split each into a list of words and a list of tags.
2. Split the data into 80% training and 20% testing.
3. Create a word tokenizer (using the training data only) and a tag tokenizer, and convert both to numbers.
4. Pad all sequences to the longest training sentence length.
5. Build the Encoder with an Embedding layer (128) and an LSTM layer (128 units) that returns its outputs and states.
6. Build the Decoder with an LSTM layer (128 units) that starts from the encoder states, followed by a Dense softmax layer that predicts
   a tag for each word.
7. Compile with the Adam optimizer and sparse categorical cross-entropy loss, then train for 5 epochs with batch size 32 and 10% validation data.
8. Evaluate on the test data, take the highest-probability tag for each word, and compare predicted and actual tags.

Result:

Thus, an Encoder–Decoder LSTM model for Part-of-Speech tagging was successfully developed and trained using the Penn Treebank corpus. 
The trained model successfully predicted the appropriate Part-of-Speech tags for the given sentences.
"""

import nltk, numpy as np
from nltk.corpus import treebank
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
nltk.download("treebank")

sents = treebank.tagged_sents()
X = [[w for w, t in s] for s in sents]
Y = [[t for w, t in s] for s in sents]
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)

wtok = Tokenizer(lower=True, oov_token="<UNK>"); wtok.fit_on_texts(Xtr)     
ttok = Tokenizer(lower=False, filters="");       ttok.fit_on_texts(Ytr)
wv, tv = len(wtok.word_index) + 1, len(ttok.word_index) + 1
max_len = max(len(s) for s in Xtr)

def pad(tok, data):
    return pad_sequences(tok.texts_to_sequences(data), maxlen=max_len, padding="post", truncating="post")
Xtr_p, Xte_p, Ytr_p, Yte_p = pad(wtok, Xtr), pad(wtok, Xte), pad(ttok, Ytr), pad(ttok, Yte)
print("Vocab:", wv, "| Tags:", tv, "| Max len:", max_len, "| Input:", Xtr_p.shape)

inp = Input((max_len,))
enc_out, h, c = LSTM(128, return_sequences=True, return_state=True)(Embedding(wv, 128, mask_zero=True)(inp))
dec_out = LSTM(128, return_sequences=True)(enc_out, initial_state=[h, c])
model = Model(inp, Dense(tv, activation="softmax")(dec_out))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(Xtr_p, Ytr_p[..., None], validation_split=0.1, epochs=5, batch_size=32)
loss, acc = model.evaluate(Xte_p, Yte_p[..., None])
print("Test Loss:", loss, "| Test Accuracy:", acc)

id2tag = {v: k for k, v in ttok.word_index.items()}
pred = model.predict(Xte_p).argmax(-1)
i = 782                                                         
print("\nWords:", Xte[i], "\nActual:", Yte[i], "\nPredicted:", [id2tag.get(t, "UNK") for t in pred[i][:len(Xte[i])]])

