"""
Aim: To build a neural network recommendation system using embeddings that suggests products to customers from their past
purchases in sales data.

Algorithm:

1. Load the sales data, remove rows with missing values, and keep only completed orders with quantity greater than 0.
2. Make unique customer-product pairs (purchases) and give each customer and product a number. These pairs get label 1.
3. For every customer, pick up to 4 random products they did not buy as negative samples with label 0.
4. Split the data into 80% training and 20% validation using stratified sampling.
5. Build the model: Embedding (32) for the customer and for the product, join them, then Dense (64), Dropout (0.2), Dense (32) and
   a Dense sigmoid output.
6. Compile with the Adam optimizer, binary cross-entropy loss and AUC as the metric, then train for 30 epochs with batch size 64.
7. For a customer, score all products they have not bought and sort the scores from highest to lowest.
8. Display the top 5 products with their names, categories and recommendation scores.

Result:

Thus, an embedding-based product recommendation model was successfully developed and trained using the sales data. The trained
model successfully recommended the top products for the given customer.
"""

import numpy as np, tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Embedding, Concatenate
tf.keras.utils.set_random_seed(42)
import pandas as pd
from sklearn.model_selection import train_test_split

cols = ["customer_id", "product_id", "product_name", "product_category", "quantity", "order_status"]
df = pd.read_csv("sales.csv").dropna(subset=cols)
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df = df[(df["order_status"].astype(str).str.strip().str.lower() == "completed") & (df["quantity"] > 0)]

pos = df[["customer_id", "product_id"]].drop_duplicates()
pos["u"], users = pd.factorize(pos["customer_id"])
pos["i"], items = pd.factorize(pos["product_id"])
nu, ni = len(users), len(items)

rng = np.random.default_rng(42)
bought = pos.groupby("u")["i"].apply(set)
neg = [(u, i) for u, s in bought.items()
       for i in rng.choice(list(set(range(ni)) - s), min(4, ni - len(s)), replace=False)]
data = pd.concat([pos[["u", "i"]].assign(y=1.0), pd.DataFrame(neg, columns=["u", "i"]).assign(y=0.0)])
tr, va = train_test_split(data, test_size=0.2, random_state=42, stratify=data["y"])
xy = lambda d: ([d["u"].values, d["i"].values], d["y"].values)

u_in, i_in = Input((1,)), Input((1,))
emb = lambda n, x: Flatten()(Embedding(n, 32)(x))
x = Dense(64, activation="relu")(Concatenate()([emb(nu, u_in), emb(ni, i_in)]))
x = Dense(32, activation="relu")(Dropout(0.2)(x))
model = Model([u_in, i_in], Dense(1, activation="sigmoid")(x))
model.compile("adam", "binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
model.fit(*xy(tr), validation_data=xy(va), epochs=30, batch_size=64)

prod = df.drop_duplicates("product_id").set_index("product_id")
def recommend(cust, n=5):
    u = users.get_loc(cust)
    cand = np.array(sorted(set(range(ni)) - bought.get(u, set())))
    if len(cand) == 0: return pd.DataFrame()
    s = model.predict([np.full(len(cand), u), cand], verbose=0).ravel()
    top = np.argsort(s)[::-1][:n]
    out = prod.loc[items[cand[top]], ["product_name", "product_category"]].reset_index()
    out["score"] = s[top].round(4)
    return out

c = users[10]
print("Customer:", c); print(recommend(c).to_string(index=False))