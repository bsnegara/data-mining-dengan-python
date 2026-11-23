import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.tree import DecisionTreeClassifier
from siapkan import muat, pipa


X, y = muat()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                  random_state=7)
print("model            AUC CV (10 lipatan)")
print(f"aturan recency   {roc_auc_score(yl, Xl['recency_hari']):.3f}"
      "  (tanpa pelatihan)")                                     #<1>
kandidat = {
    "pohon (d=5)": pipa(DecisionTreeClassifier(max_depth=5,
                                               random_state=0), False),
    "hutan acak": pipa(RandomForestClassifier(300, max_features=0.2,
                       min_samples_leaf=5, random_state=0, n_jobs=-1),
                       False),
    "logistik": pipa(LogisticRegression(max_iter=2000))}
cv = StratifiedKFold(10, shuffle=True, random_state=0)           #<2>
for nama, m in kandidat.items():
    s = cross_val_score(m, Xl, yl, cv=cv, scoring="roc_auc")
    print(f"{nama:16s} {s.mean():.3f} +/- {s.std():.3f}")


def presisi_atas(y, p, q=0.0385):
    k = int(round(q * len(y)))
    return y[np.argsort(-p)[:k]].mean()


print("\ndata uji (dikunci sampai sekarang):")
yu_ = yu.to_numpy()
skor = {"aturan recency": Xu["recency_hari"].to_numpy() / 90,  #<3>
        "logistik": kandidat["logistik"].fit(Xl, yl).predict_proba(
            Xu)[:, 1]}
for nama, p in skor.items():
    print(f"{nama:16s} AUC {roc_auc_score(yu_, p):.3f}  presisi top "
          f"3,85% {presisi_atas(yu_, p):.3f}")

rng = np.random.default_rng(0)                                   #<4>
auc, pre = [], []
for _ in range(2000):
    i = rng.integers(0, len(yu_), len(yu_))
    auc.append(roc_auc_score(yu_[i], p[i]))
    pre.append(presisi_atas(yu_[i], p[i]))
print("selang 95% logistik: AUC "
      f"[{np.percentile(auc, 2.5):.3f}, {np.percentile(auc, 97.5):.3f}]"
      f", presisi [{np.percentile(pre, 2.5):.3f}, "
      f"{np.percentile(pre, 97.5):.3f}]")
