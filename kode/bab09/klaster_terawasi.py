import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class KlasterTerawasi(ClassifierMixin, BaseEstimator):
    def __init__(self, m=3):
        self.m = m

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        pusat, label = [], []
        for c in self.classes_:                                  #<1>
            km = KMeans(self.m, n_init=10, random_state=0).fit(X[y == c])
            pusat.append(km.cluster_centers_)
            label += [c] * self.m
        self.pusat_, self.label_ = np.vstack(pusat), np.array(label)
        return self

    def predict(self, X):                                        #<2>
        d = ((np.asarray(X)[:, None, :] - self.pusat_) ** 2).sum(-1)
        return self.label_[d.argmin(1)]


DATA = "kode/data/"
biji = pd.read_csv(DATA + "mutu_biji.csv")
rasa = ["aroma", "flavor", "aftertaste", "acidity", "body", "balance",
        "overall"]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for m in [1, 2, 4, 8, 16]:                                       #<3>
    s = cross_val_score(make_pipeline(StandardScaler(),
                                      KlasterTerawasi(m)),
                        biji[rasa], biji["asal"], cv=cv).mean()
    print(f"{m:2d} klaster per kelas ({4 * m:3d} prototipe): {s:.3f}")
s = cross_val_score(make_pipeline(StandardScaler(),
                                  KNeighborsClassifier(25)),
                    biji[rasa], biji["asal"], cv=cv).mean()
print(f"kNN k=25 (1200 objek latih)       : {s:.3f}")
