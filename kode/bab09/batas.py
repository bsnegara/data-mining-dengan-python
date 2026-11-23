import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

DATA = "kode/data/"
biji = pd.read_csv(DATA + "mutu_biji.csv")
X = biji[["acidity", "body"]].to_numpy()
y = biji["asal"].map({"Gayo": 0, "Kerinci": 1, "Toraja": 2, "Flores": 3})
gx, gy = np.meshgrid(np.linspace(5.8, 8.8, 61), np.linspace(6.0, 8.8, 57))
kisi = np.c_[gx.ravel(), gy.ravel()]
for k in [1, 25]:
    m = KNeighborsClassifier(k).fit(X + np.random.default_rng(0).normal(
        0, 0.02, X.shape), y)                                    #<1>
    pd.DataFrame({"x": kisi[:, 0], "y": kisi[:, 1],
                  "c": m.predict(kisi)}).to_csv(
        f"gambar/bab09/batas{k}.dat", sep=" ", index=False)
sampel = biji.sample(240, random_state=1)
geser = np.random.default_rng(2).uniform(-0.09, 0.09, (240, 2))  #<2>
pd.DataFrame({"x": sampel["acidity"] + geser[:, 0],
              "y": sampel["body"] + geser[:, 1],
              "c": y[sampel.index]}).to_csv("gambar/bab09/titik.dat",
                                             sep=" ", index=False)
print("selesai")
