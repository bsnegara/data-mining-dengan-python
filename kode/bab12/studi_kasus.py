import lightgbm as lgb
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from siapkan import KATEGORIK, muat, pra_pohon

X, y, num = muat(tren=True)                                      #<1>
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                  random_state=7)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
logistik = make_pipeline(ColumnTransformer(
    [("num", StandardScaler(), num),
     ("kat", OneHotEncoder(handle_unknown="ignore"), KATEGORIK)]),
    LogisticRegression(C=3.162, max_iter=3000))
hutan = make_pipeline(pra_pohon(num), RandomForestClassifier(
    500, min_samples_leaf=5, max_features=0.3, random_state=0,
    n_jobs=-1))
ruang = {"lgbmclassifier__num_leaves": [4, 7, 15, 31],          #<2>
         "lgbmclassifier__learning_rate": [0.01, 0.02, 0.05],
         "lgbmclassifier__n_estimators": [200, 400, 800],
         "lgbmclassifier__min_child_samples": [20, 50, 100],
         "lgbmclassifier__colsample_bytree": [0.5, 0.8, 1.0],
         "lgbmclassifier__reg_lambda": [0, 1, 10]}
cari = RandomizedSearchCV(make_pipeline(pra_pohon(num), lgb.LGBMClassifier(
    subsample=0.8, subsample_freq=1, random_state=0, verbose=-1,
    n_jobs=1)),
    ruang, n_iter=15, cv=cv, scoring="roc_auc", random_state=0,
    n_jobs=-1).fit(Xl, yl)
gbm = cari.best_estimator_
print("LightGBM terbaik:", {k.split("__")[1]: v
                            for k, v in cari.best_params_.items()})
tumpuk = StackingClassifier(                                     #<3>
    [("log", logistik), ("gbm", gbm)],
    final_estimator=LogisticRegression(), cv=cv, n_jobs=-1)

print("\nmodel           AUC-CV  AUC-uji  AP-uji  Brier   laba(jt)")
p_uji = {}
for nama, m in [("logistik", logistik), ("random forest", hutan),
                ("LightGBM", gbm), ("tumpuk", tumpuk)]:
    auc_cv = cross_val_score(m, Xl, yl, cv=cv, scoring="roc_auc",
                             n_jobs=-1).mean()
    p = m.fit(Xl, yl).predict_proba(Xu)[:, 1]
    p_uji[nama] = p
    sasar = p >= 15 / 54                                         #<4>
    laba = (54_000 * yu[sasar].sum() - 15_000 * sasar.sum()) / 1e6
    print(f"{nama:14s} {auc_cv:7.3f} {roc_auc_score(yu, p):8.3f} "
          f"{average_precision_score(yu, p):7.3f} "
          f"{brier_score_loss(yu, p):7.4f} {laba:8.1f}")

rng = np.random.default_rng(0)                                   #<5>
yv = yu.to_numpy()
beda = []
for _ in range(1000):
    i = rng.integers(0, len(yv), len(yv))
    beda.append(roc_auc_score(yv[i], p_uji["tumpuk"][i])
                - roc_auc_score(yv[i], p_uji["logistik"][i]))
lo, hi = np.percentile(beda, [2.5, 97.5])
print(f"\nselisih AUC tumpuk - logistik: {np.mean(beda):+.4f} "
      f"[{lo:+.4f}, {hi:+.4f}]")
r = np.corrcoef(p_uji["logistik"], p_uji["LightGBM"])[0, 1]
print(f"korelasi skor logistik vs LightGBM: {r:.3f}")
