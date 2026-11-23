import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, "kode/bab07")
from siapkan import muat, pipa  # noqa: E402

X, y = muat()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                  random_state=7)
biasa = pipa(LogisticRegression(max_iter=2000)).fit(Xl, yl)
seimbang = pipa(LogisticRegression(max_iter=2000,                #<1>
                                   class_weight="balanced")).fit(Xl, yl)
for nama, m in [("biasa", biasa), ("balanced", seimbang)]:
    p = m.predict_proba(Xu)[:, 1]
    print(f"{nama:8s}: rata-rata peluang {p.mean():.3f} "
          f"(porsi churn {yu.mean():.3f})")

p = biasa.predict_proba(Xu)[:, 1]
ambang = np.linspace(0.05, 0.9, 86)
f1 = [f1_score(yu, p >= t) for t in ambang]                      #<2>
laba = [((p >= t) & (yu == 1)).sum() * 39_000
        - ((p >= t) & (yu == 0)).sum() * 15_000 for t in ambang]  #<3>
for nama, t in [("bawaan", 0.5), ("F1 maksimum", ambang[np.argmax(f1)]),
                ("laba maksimum", ambang[np.argmax(laba)]),
                ("teoretis c_FP/(c_FP+c_FN)", 15 / 54)]:
    tanda = p >= t
    lb = tanda[yu == 1].sum() * 39_000 - tanda[yu == 0].sum() * 15_000
    print(f"{nama:26s} t={t:.3f}: ditandai {tanda.sum():4d}, "
          f"laba Rp{lb / 1e6:5.2f} jt")
