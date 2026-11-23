import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from siapkan import muat, pipa

X, y = muat()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                  random_state=7)
model = {"logistik": pipa(LogisticRegression(max_iter=2000)),
         "pohon": pipa(DecisionTreeClassifier(max_depth=5,
                                              random_state=0), False),
         "hutan": pipa(RandomForestClassifier(300, min_samples_leaf=5,
                       random_state=0, n_jobs=-1), False)}
grid = np.linspace(0, 1, 101)
roc, pr = {"fpr": grid}, {"recall": grid}
for nama, m in model.items():
    p = m.fit(Xl, yl).predict_proba(Xu)[:, 1]
    fpr, tpr, _ = roc_curve(yu, p)                               #<1>
    pre, rec, _ = precision_recall_curve(yu, p)                  #<2>
    print(f"{nama:9s} AUC {roc_auc_score(yu, p):.3f}  "
          f"AP {average_precision_score(yu, p):.3f}")
    roc[nama] = np.interp(grid, fpr, tpr)
    urut = np.argsort(rec)                                       #<3>
    pr[nama] = np.interp(grid, rec[urut], np.maximum.accumulate(
        pre[urut][::-1])[::-1])
print(f"garis dasar PR = porsi churn = {yu.mean():.3f}")
pd.DataFrame(roc).round(4).to_csv("gambar/bab07/roc.dat", sep=" ",
                                  index=False)
pd.DataFrame(pr).round(4).to_csv("gambar/bab07/pr.dat", sep=" ",
                                 index=False)
