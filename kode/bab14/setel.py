import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from siapkan import muat, pra

df, X, y = muat()
pipa = make_pipeline(pra(), SVC(kernel="rbf"))
kisi = {"svc__C": np.logspace(-2, 3, 6),                         #<1>
        "svc__gamma": np.logspace(-4, 1, 6)}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
g = GridSearchCV(pipa, kisi, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X, y)
tab = pd.DataFrame(g.cv_results_["mean_test_score"].reshape(6, 6),
                   index=[f"C={c:g}" for c in kisi["svc__C"]],
                   columns=[f"{v:g}" for v in kisi["svc__gamma"]])
print("AUC CV (baris C, kolom gamma):")
print(tab.round(3).to_string())
b = g.best_params_
print(f"terbaik: C={b['svc__C']:g}, gamma={b['svc__gamma']:g}, "
      f"AUC {g.best_score_:.3f}")
with open("gambar/bab14/kisi.dat", "w") as f:                    #<2>
    for i, c in enumerate(kisi["svc__C"]):
        for j, v in enumerate(kisi["svc__gamma"]):
            f.write(f"{j} {i} {tab.iloc[i, j]:.3f}\n")
        f.write("\n")
