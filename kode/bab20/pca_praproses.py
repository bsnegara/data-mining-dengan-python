import sys; sys.path.insert(0, "kode/bab15")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from siapkan import muat

df, X = muat()                                                   #<1>
y = df.churn
pca = make_pipeline(StandardScaler(), PCA()).fit(X)
rasio = pca[-1].explained_variance_ratio_
print("porsi varians :", rasio.round(3))
print("kumulatif     :", rasio.cumsum().round(3))

for k in [1, 2, 3, 5, 7]:                                        #<2>
    model = make_pipeline(StandardScaler(), PCA(n_components=k),
                          LogisticRegression(max_iter=1000))
    auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(f"PCA {k} komponen -> logistik: AUC {auc.mean():.3f}")
