import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA = "kode/data/"
df = pd.read_csv(DATA + "pelanggan_bersih.csv").dropna()
fitur = ["frekuensi_90h", "recency_hari", "porsi_promo",
         "keluhan_90h", "jarak_km", "nilai_90h"]
X = StandardScaler().fit_transform(df[fitur])
y = df["churn"].values
print("porsi churn:", y.mean().round(3))
Xl, Xu, yl, yu = train_test_split(X, y, test_size=.3,
                                  stratify=y, random_state=1)

def smote(X, y, k=5, rng=np.random.default_rng(0)):              #<1>
    Xm = X[y == 1]
    perlu = (y == 0).sum() - len(Xm)
    jarak = ((Xm[:, None, :] - Xm[None, :, :]) ** 2).sum(-1)
    tetangga = np.argsort(jarak, axis=1)[:, 1:k + 1]
    i = rng.integers(0, len(Xm), perlu)
    j = tetangga[i, rng.integers(0, k, perlu)]
    lam = rng.random((perlu, 1))
    baru = Xm[i] + lam * (Xm[j] - Xm[i])                         #<2>
    return np.vstack([X, baru]), np.r_[y, np.ones(perlu, int)]

rng = np.random.default_rng(0)
idx0, idx1 = np.where(yl == 0)[0], np.where(yl == 1)[0]
bawah = np.r_[rng.choice(idx0, len(idx1), replace=False), idx1]  #<3>
atas = np.r_[idx0, rng.choice(idx1, len(idx0), replace=True)]
cara = {"asli": (Xl, yl, None), "undersampling": (Xl[bawah], yl[bawah], None),
        "oversampling": (Xl[atas], yl[atas], None),
        "SMOTE": (*smote(Xl, yl), None),
        "bobot kelas": (Xl, yl, "balanced")}                     #<4>
for nama, (Xa, ya, bobot) in cara.items():
    m = LogisticRegression(class_weight=bobot, max_iter=1000).fit(Xa, ya)
    p = m.predict(Xu)
    print(f"{nama:<14} n={len(ya):>5}  presisi={precision_score(yu, p):.2f}"
          f"  recall={recall_score(yu, p):.2f}")
