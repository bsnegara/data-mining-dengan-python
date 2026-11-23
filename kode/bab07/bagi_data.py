import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from siapkan import muat, pipa

X, y = muat()
m = pipa(LogisticRegression(max_iter=2000))

akurasi = []                                                     #<1>
for benih in range(20):
    Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3,
                                      random_state=benih)
    akurasi.append(m.fit(Xl, yl).score(Xu, yu))
akurasi = np.array(akurasi)
print(f"hold-out 20 kali : {akurasi.min():.3f} s.d. "
      f"{akurasi.max():.3f} (rata-rata {akurasi.mean():.3f})")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  #<2>
skor = cross_val_score(m, X, y, cv=cv)
print(f"CV 10 lipatan    : {skor.mean():.3f} +/- {skor.std():.3f}")

for nama, pemecah in [("KFold", KFold(10, shuffle=True,          #<3>
                                      random_state=3)),
                      ("StratifiedKFold", cv)]:
    porsi = [y.iloc[iu].mean() for _, iu in pemecah.split(X, y)]
    print(f"{nama:16s} porsi churn per lipatan: "
          f"{min(porsi):.3f} s.d. {max(porsi):.3f}")
