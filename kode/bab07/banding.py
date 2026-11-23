import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from siapkan import muat, pipa

X, y = muat()
a = pipa(LogisticRegression(max_iter=2000))
b = pipa(RandomForestClassifier(300, min_samples_leaf=5,
                                random_state=0, n_jobs=-1), False)

d = []                                                           #<1>
for r in range(5):
    cv = StratifiedKFold(2, shuffle=True, random_state=r)
    beda = []
    for il, iu in cv.split(X, y):
        sa = roc_auc_score(y.iloc[iu], clone(a).fit(X.iloc[il],
                           y.iloc[il]).predict_proba(X.iloc[iu])[:, 1])
        sb = roc_auc_score(y.iloc[iu], clone(b).fit(X.iloc[il],
                           y.iloc[il]).predict_proba(X.iloc[iu])[:, 1])
        beda.append(sa - sb)
    d.append(beda)
d = np.array(d)
s2 = ((d - d.mean(1, keepdims=True)) ** 2).sum(1)                #<2>
t = d[0, 0] / np.sqrt(s2.mean())
p = 2 * stats.t.sf(abs(t), df=5)
print(f"selisih AUC rata-rata (log - hutan): {d.mean():+.4f}")
print(f"uji 5x2cv: t = {t:.3f}, p = {p:.3f}")

Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                  random_state=7)
ba = a.fit(Xl, yl).predict(Xu) == yu                             #<3>
bb = b.fit(Xl, yl).predict(Xu) == yu
n01, n10 = int((ba & ~bb).sum()), int((~ba & bb).sum())
chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
print(f"McNemar: hanya log benar {n01}, hanya hutan benar {n10}")
print(f"         chi2 = {chi:.3f}, p = {stats.chi2.sf(chi, 1):.3f}")
