import time
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score)
from siapkan import muat, pipa

X, y = muat()
cv = StratifiedKFold(5, shuffle=True, random_state=0)
hutan = pipa(RandomForestClassifier(200, random_state=0, n_jobs=-1),
             skala=False)
awalan = "randomforestclassifier__"
kisi = {awalan + "max_features": [0.2, 0.4, 0.6],                #<1>
        awalan + "min_samples_leaf": [1, 5, 20, 50]}
t = time.time()
g = GridSearchCV(hutan, kisi, cv=cv, scoring="roc_auc").fit(X, y)
print(f"grid  : {len(g.cv_results_['params'])} kombinasi, "
      f"AUC {g.best_score_:.4f}, {time.time() - t:.0f} dtk")
print("        terbaik:", {k.replace(awalan, ""): v
                           for k, v in g.best_params_.items()})

sebaran = {awalan + "max_features": [0.1, 0.2, 0.3, 0.4, 0.6, 0.8],
           awalan + "min_samples_leaf": randint(1, 80)}          #<2>
t = time.time()
r = RandomizedSearchCV(hutan, sebaran, n_iter=12, cv=cv,
                       scoring="roc_auc", random_state=0).fit(X, y)
print(f"acak  : 12 kombinasi, AUC {r.best_score_:.4f}, "
      f"{time.time() - t:.0f} dtk")
print("        terbaik:", {k.replace(awalan, ""): v
                           for k, v in r.best_params_.items()})

dalam = GridSearchCV(hutan, kisi, cv=3, scoring="roc_auc")       #<3>
luar = cross_val_score(dalam, X, y, cv=cv, scoring="roc_auc")
print(f"nested CV: AUC {luar.mean():.4f} +/- {luar.std():.4f}")
