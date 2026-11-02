"""
Aim: To build an Encoder–Decoder LSTM model that translates English sentences into Tamil after training on paired sentences.

Algorithm:

1. Read the English–Tamil sentence pairs from the tab-separated file, and remove empty lines and duplicate pairs.
2. Add <start> and <end> tokens to every Tamil sentence.
3. Create separate tokenizers for English and Tamil, convert the sentences to numbers, and pad them to the maximum length.
4. Make the decoder input from the Tamil sequence without its last token, and the decoder output from the sequence 
   without its first token (shifted by one).
5. Build the Encoder with an Embedding layer (64) and an LSTM layer (128 units), and pass its hidden and cell states to the Decoder.
6. Build the Decoder with an Embedding layer (64), an LSTM layer (128 units) and a Dense softmax layer.
7. Compile with the Adam optimizer and sparse categorical cross-entropy loss, then train for 100 epochs with batch size 32.
8. To translate, start with <start>, predict one Tamil word at a time, and stop when <end> is generated.

Result:

Thus, an LSTM-based English-to-Tamil neural machine translation model was successfully developed and trained using paired sentence 
data. The trained model successfully generated Tamil translations for the given English test sentences.
"""

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

pairs = []
for line in open(r"eng_tamil_pairs.txt", encoding="utf-8"):
    p = line.strip().split("\t", 1)
    if len(p) == 2: pairs.append((p[0].strip(), p[1].strip()))
print("Pairs:", len(pairs))
pairs = list(dict.fromkeys(pairs))                              
print("Unique pairs:", len(pairs))
english = [e for e, t in pairs]
tamil = [f"<start> {t} <end>" for e, t in pairs]

def tokenize(texts):
    tok = Tokenizer(filters=""); tok.fit_on_texts(texts)
    seq = tok.texts_to_sequences(texts); L = max(len(s) for s in seq)
    return tok, pad_sequences(seq, maxlen=L, padding="post"), L
et, eng, max_eng = tokenize(english)
tt, tam, max_tam = tokenize(tamil)
ev, tv = len(et.word_index) + 1, len(tt.word_index) + 1
print("Max len:", max_eng, max_tam, "| Vocab:", ev, tv)

dec_in_data, dec_out_data = tam[:, :-1], tam[:, 1:]             

enc_in = Input((max_eng,))
_, h, c = LSTM(128, return_state=True)(Embedding(ev, 64, mask_zero=True)(enc_in))   
dec_in = Input((max_tam - 1,))
dec = LSTM(128, return_sequences=True)(Embedding(tv, 64, mask_zero=True)(dec_in), initial_state=[h, c])  
model = Model([enc_in, dec_in], Dense(tv, activation="softmax")(dec))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit([eng, dec_in_data], dec_out_data[..., None], epochs=100, batch_size=32, verbose=1)

rev = {i: w for w, i in tt.word_index.items()}
start, end = tt.word_index["<start>"], tt.word_index["<end>"]
def translate(sentence):
    src = pad_sequences(et.texts_to_sequences([sentence]), maxlen=max_eng, padding="post")
    tgt = [start]
    for _ in range(max_tam - 1):                                
        t = pad_sequences([tgt], maxlen=max_tam - 1, padding="post")
        nxt = model.predict([src, t], verbose=0)[0, len(tgt) - 1].argmax()
        if nxt in (0, end): break
        tgt.append(nxt)
    return " ".join(rev[i] for i in tgt[1:])

for s in ["Get down.", "Goodbye and take care.", "Let's meet again soon.", "Good morning.", "Thank you.", "How are you?", "hospital"]:
    print("\nEnglish :", s, "\nTamil   :", translate(s))