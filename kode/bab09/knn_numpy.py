import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

DATA = "kode/data/"
biji = pd.read_csv(DATA + "mutu_biji.csv", parse_dates=["tanggal"])
rasa = ["aroma", "flavor", "aftertaste", "acidity", "body", "balance",
        "overall"]
latih = biji[biji["tanggal"] < "2025-01-01"]
uji = biji[biji["tanggal"] >= "2025-01-01"]
Xl, yl = latih[rasa].to_numpy(), latih["asal"].to_numpy()
Xu, yu = uji[rasa].to_numpy(), uji["asal"].to_numpy()


def knn(Xl, yl, Xu, k):
    d = ((Xu[:, None, :] - Xl[None, :, :]) ** 2).sum(-1)         #<1>
    tetangga = np.argsort(d, axis=1)[:, :k]                      #<2>
    pred = []
    for baris in yl[tetangga]:                                   #<3>
        nilai, cacah = np.unique(baris, return_counts=True)
        pred.append(nilai[np.argmax(cacah)])
    return np.array(pred)


for k in [1, 5, 25]:
    sendiri = (knn(Xl, yl, Xu, k) == yu).mean()
    sk = KNeighborsClassifier(k).fit(Xl, yl).score(Xu, yu)       #<4>
    print(f"k={k:2d}: NumPy {sendiri:.3f}  scikit-learn {sk:.3f}")
