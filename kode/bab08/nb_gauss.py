import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

DATA = "kode/data/"
df = pd.read_csv(DATA + "pelanggan.csv")
fitur = ["recency_hari", "frekuensi_90h", "nilai_90h", "keluhan_90h",
         "ragam_menu", "porsi_promo"]
X, y = df[fitur], df["churn"]
nb = GaussianNB().fit(X, y)
tab = pd.DataFrame(nb.theta_, columns=fitur, index=["setia", "churn"])
print("rata-rata per kelas:")                                    #<1>
print(tab.T.round(2).to_string())
print("simpangan baku recency:", np.sqrt(nb.var_[:, 0]).round(2))

for nama, Xu in [("asli", X),                                    #<2>
                 ("log1p", np.log1p(X))]:
    s = cross_val_score(GaussianNB(), Xu, y, cv=5, scoring="roc_auc")
    print(f"AUC CV, {nama:6s}: {s.mean():.3f}")

r = df["recency_hari"].to_numpy()                                #<3>
grid = np.arange(0, 90, 5.0) + 2.5
keluar = {"x": grid}
for c, nama in [(0, "setia"), (1, "churn")]:
    m, s = nb.theta_[c, 0], np.sqrt(nb.var_[c, 0])
    keluar["g_" + nama] = np.exp(-(grid - m) ** 2 / (2 * s * s)) / (
        s * np.sqrt(2 * np.pi))
    h, _ = np.histogram(r[y == c], bins=np.arange(0, 91, 5), density=True)
    keluar["h_" + nama] = h
pd.DataFrame(keluar).round(5).to_csv("gambar/bab08/recency.dat", sep=" ",
                                     index=False)
