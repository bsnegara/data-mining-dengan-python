import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from kartu import kartu, y                                       #<1>

Xl, Xu, yl, yu = train_test_split(kartu, y, test_size=0.4, stratify=y,
                                  random_state=1)
k = [int(kartu[c].max()) + 1 for c in kartu.columns]             #<2>
hasil = []
for n in [20, 40, 80, 160, 320, 640, 1280, 2560, 4800]:
    a, b = [], []
    for r in range(30):                                          #<3>
        i = Xl.sample(n, random_state=r).index
        if yl[i].nunique() < 2:
            continue
        nb = CategoricalNB(alpha=1, min_categories=k).fit(Xl.loc[i], yl[i])
        lg = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                           LogisticRegression(max_iter=2000)).fit(
            Xl.loc[i], yl[i])
        a.append(roc_auc_score(yu, nb.predict_proba(Xu)[:, 1]))
        b.append(roc_auc_score(yu, lg.predict_proba(Xu)[:, 1]))
    hasil.append((n, np.mean(a), np.mean(b)))
tab = pd.DataFrame(hasil, columns=["n", "nb", "log"])
print(tab.round(3).to_string(index=False))
tab.to_csv("gambar/bab08/kurva_belajar.dat", sep=" ", index=False)
