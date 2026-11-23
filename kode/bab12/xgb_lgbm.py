import time
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from siapkan import muat

X, y, num = muat()
X = X.astype({k: "category" for k in ["kota", "tingkat",
                                      "jenis_kelamin"]})         #<1>
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                  random_state=7)
Xa, Xv, ya, yv = train_test_split(Xl, yl, test_size=0.2, stratify=yl,
                                  random_state=1)                #<2>
model = {
    "XGBoost": xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        enable_categorical=True, tree_method="hist",
        early_stopping_rounds=100, eval_metric="logloss",
        random_state=0),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=15,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=40, random_state=0, verbose=-1),
    "HistGB sklearn": HistGradientBoostingClassifier(
        max_iter=2000, learning_rate=0.03, max_leaf_nodes=15,
        categorical_features="from_dtype", early_stopping=True,
        validation_fraction=0.2, n_iter_no_change=100,
        random_state=0)}
for nama, m in model.items():
    t0 = time.perf_counter()
    if nama == "XGBoost":
        m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)       #<3>
        n = m.best_iteration + 1
    elif nama == "LightGBM":
        m.fit(Xa, ya, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        n = m.best_iteration_
    else:
        m.fit(Xl, yl)
        n = m.n_iter_
    dt = time.perf_counter() - t0
    auc = roc_auc_score(yu, m.predict_proba(Xu)[:, 1])
    print(f"{nama:15s} AUC uji {auc:.3f}  pohon {n:4d}  "
          f"waktu {dt:4.1f} dtk")
