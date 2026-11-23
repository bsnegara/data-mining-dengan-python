"""Data gambar Bab 5."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

pel = pd.read_csv("kode/data/pelanggan_bersih.csv")
x = pel["nilai_90h"] / 1000
for nama, v, rng in [("z", (x - x.mean()) / x.std(), (-2, 8)),
                     ("log", np.log1p(x), (0, 9)),
                     ("yeo", PowerTransformer().fit_transform(x.to_frame()).ravel(), (-3, 4.5))]:
    h, t = np.histogram(v, bins=30, range=rng)
    pd.DataFrame({"x": t[:-1], "n": h}).to_csv(f"gambar/bab05/hist_{nama}.dat", sep=" ", index=False)
