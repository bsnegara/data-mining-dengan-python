import numpy as np
import pandas as pd

R = pd.DataFrame([[5, 3, 0, 1, 0, 2],                            #<1>
                  [4, 0, 0, 1, 0, 1],
                  [1, 1, 0, 5, 4, 0],
                  [0, 1, 5, 4, 0, 0],
                  [4, 2, 0, 0, 1, 0]],
                 index=["Ani", "Budi", "Citra", "Dedi", "Eka"],
                 columns=["gula aren", "pandan", "V60", "americano",
                          "espresso", "roti"])
X = R.to_numpy(float)
norma = np.linalg.norm(X, axis=1)
S = X @ X.T / np.outer(norma, norma)                             #<2>
print("kemiripan kosinus antarpengguna:")
print(pd.DataFrame(S, R.index, R.index).round(3).to_string())
u = 1                                                            #<3>
tetangga = [i for i in np.argsort(-S[u]) if i != u][:2]
print("dua tetangga Budi:", [R.index[i] for i in tetangga])
for j in np.where(X[u] == 0)[0]:                                 #<4>
    w = S[u, tetangga]
    pred = (w * X[tetangga, j]).sum() / w.sum()
    print(f"  ramalan Budi untuk {R.columns[j]:9s}: {pred:.2f}")
