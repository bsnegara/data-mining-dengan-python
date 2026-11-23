import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from siapkan import muat, pra_pohon

X, y, num = muat()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                  random_state=7)
pra = pra_pohon(num).fit(Xl)
Zl, Zu = pra.transform(Xl), pra.transform(Xu)
hasil = []
print("AUC uji setelah B pohon; B =  1      5      20     100    500")
for mf in ["sqrt", 0.5, None]:                                   #<1>
    rf = RandomForestClassifier(500, max_features=mf, oob_score=True,
                                min_samples_leaf=3, random_state=0,
                                n_jobs=-1).fit(Zl, yl)
    P = np.array([t.predict_proba(Zu)[:, 1] for t in rf.estimators_])
    kum = np.cumsum(P, axis=0) / np.arange(1, 501)[:, None]      #<2>
    auc = [roc_auc_score(yu, kum[b - 1]) for b in (1, 5, 20, 100, 500)]
    hasil.append([roc_auc_score(yu, kum[b - 1]) for b in range(1, 501)])
    print(f"max_features={mf!s:5s}: " + "  ".join(f"{a:.3f}" for a in auc))
with open("gambar/bab12/jumlah_pohon.dat", "w") as f:
    f.write("b sqrt setengah semua\n")
    for b in list(range(1, 20)) + list(range(20, 501, 10)):
        f.write(f"{b} " + " ".join(f"{h[b - 1]:.4f}" for h in hasil)
                + "\n")
