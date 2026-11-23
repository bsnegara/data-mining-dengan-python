import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from siapkan import muat, pra_pohon

X, y, num = muat()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                  random_state=7)
rng = np.random.default_rng(0)
idx = rng.integers(0, len(Xl), len(Xl))                          #<1>
print(f"satu sampel bootstrap: {len(np.unique(idx)) / len(Xl):.1%} "
      "baris unik (teori 63,2%)")

skor = {}
for nama, m in {
        "satu pohon penuh": DecisionTreeClassifier(random_state=0),
        "bagging 200 pohon": BaggingClassifier(                  #<2>
            DecisionTreeClassifier(), n_estimators=200,
            oob_score=True, random_state=0, n_jobs=-1),
        "random forest 200": RandomForestClassifier(             #<3>
            200, max_features="sqrt", oob_score=True,
            random_state=0, n_jobs=-1)}.items():
    pipa = make_pipeline(pra_pohon(num), m).fit(Xl, yl)
    p = pipa.predict_proba(Xu)[:, 1]
    oob = getattr(m, "oob_score_", float("nan"))
    print(f"{nama:18s} AUC uji {roc_auc_score(yu, p):.3f}  "
          f"akurasi OOB {oob:.3f}")
    skor[nama] = p

print("\nkorelasi prediksi antarpohon (rata-rata, 30 pohon pertama):")
for nama in ["bagging 200 pohon", "random forest 200"]:          #<4>
    m = (BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                           random_state=0)
         if nama.startswith("bagging") else
         RandomForestClassifier(30, max_features="sqrt", random_state=0))
    pipa = make_pipeline(pra_pohon(num), m).fit(Xl, yl)
    Z = pipa[0].transform(Xu)
    P = np.array([t.predict_proba(np.asarray(Z))[:, 1]
                  for t in m.estimators_])
    r = np.corrcoef(P)[np.triu_indices(30, 1)].mean()
    print(f"  {nama:18s} rho = {r:.3f}")
