import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from siapkan import muat, pipa

VOUCER, SELAMAT, LABA = 15_000, 0.30, 180_000                    #<1>
ANGGOTA = 26_000
X, y = muat()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                  random_state=7)
p = pipa(LogisticRegression(max_iter=2000)).fit(Xl, yl).predict_proba(
    Xu)[:, 1]
urut = np.argsort(-p)                                            #<2>
churn = yu.to_numpy()[urut]
n = len(churn)
k = np.arange(1, n + 1)
presisi = churn.cumsum() / k
tangkap = churn.cumsum() / churn.sum()                           #<3>
lift = presisi / churn.mean()
manfaat = SELAMAT * LABA * churn.cumsum() - VOUCER * k           #<4>
laba = manfaat / n * ANGGOTA / 1e6                               #<5>
porsi = k / n
print(f"titik impas presisi = {VOUCER / (SELAMAT * LABA):.3f}")
for q in [0.02, 0.0385, 0.05, 0.10, 0.20, 0.50, 1.0]:
    i = int(round(q * n)) - 1
    print(f"top {q:5.1%}: presisi {presisi[i]:.3f} lift {lift[i]:4.2f}"
          f" tangkap {tangkap[i]:.3f} Rp{laba[i]:6.1f} jt")
i = int(np.argmax(laba))
print(f"laba maksimum Rp{laba[i]:.1f} jt pada top {porsi[i]:.1%} "
      f"(~{porsi[i] * ANGGOTA:.0f} anggota)")
print(f"  presisi {presisi[i]:.3f}, skor ambang {p[urut][i]:.3f}")
acak = SELAMAT * LABA * yu.mean() - VOUCER                       #<6>
print(f"voucer acak: Rp{acak:,.0f} per anggota")
pilih = np.unique(np.r_[np.linspace(0, n - 1, 120).astype(int)])
pd.DataFrame({"porsi": porsi[pilih], "tangkap": tangkap[pilih],
              "lift": lift[pilih], "laba": laba[pilih]}).round(4).to_csv(
    "gambar/bab07/laba.dat", sep=" ", index=False)
