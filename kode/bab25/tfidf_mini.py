import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfVectorizer)

dok = ["kopi enak kopi wangi",                                   #<1>
       "kopi pahit pelayanan lama",
       "pelayanan ramah tempat nyaman"]
cv = CountVectorizer()
X = cv.fit_transform(dok).toarray()
kata = cv.get_feature_names_out()
print(pd.DataFrame(X, columns=kata).to_string())

n = len(dok)                                                     #<2>
df = (X > 0).sum(0)
idf = np.log((1 + n) / (1 + df)) + 1
for v in (1, 2):
    print(f"df = {v}: idf = {np.log((1 + n) / (1 + v)) + 1:.3f}  ",
          ", ".join(kata[df == v]))
W = X * idf
W = W / np.linalg.norm(W, axis=1, keepdims=True)
print("TF-IDF dokumen 1:", {k: round(float(v), 3) for k, v
                            in zip(kata, W[0]) if v > 0})

tv = TfidfVectorizer()                                           #<3>
W2 = tv.fit_transform(dok).toarray()
print("sama dengan sklearn:", np.allclose(W, W2))
cos = W2 @ W2.T
print("kemiripan kosinus:\n", cos.round(3))
