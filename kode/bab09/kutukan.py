import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
baris = []
for p in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
    X = rng.uniform(size=(1000, p))                              #<1>
    q = rng.uniform(size=(50, p))
    d = np.sqrt(((q[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    rasio = (d.min(1) / d.max(1)).mean()                         #<2>
    baris.append((p, rasio))
    print(f"p={p:3d}: jarak terdekat / terjauh = {rasio:.3f}")
pd.DataFrame(baris, columns=["p", "rasio"]).to_csv(
    "gambar/bab09/kutukan.dat", sep=" ", index=False)
