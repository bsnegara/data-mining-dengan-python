import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from siapkan import muat, pra_pohon

X, y, num = muat()
rng = np.random.default_rng(0)
X = X.assign(acak=rng.normal(size=len(X)),                       #<1>
             id_acak=rng.integers(0, 10_000, len(X)))
num = num + ["acak", "id_acak"]
X[num] = X[num].astype(float)
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                  random_state=7)
rf = make_pipeline(pra_pohon(num), RandomForestClassifier(
    300, random_state=0, n_jobs=-1)).fit(Xl, yl)
nama = rf[0].get_feature_names_out()
mdi = pd.Series(rf[-1].feature_importances_, index=nama)         #<2>
perm = permutation_importance(rf, Xu, yu, scoring="roc_auc",     #<3>
                              n_repeats=10, random_state=0, n_jobs=-1)
pi = pd.Series(perm.importances_mean, index=X.columns)
tab = pd.DataFrame({"MDI": mdi, "permutasi": pi}).sort_values(
    "permutasi", ascending=False)
print(tab.head(8).round(4).to_string())
print("...")
print(tab.loc[["acak", "id_acak"]].round(4).to_string())

pd_ = partial_dependence(rf, Xl, ["recency_hari"],               #<4>
                         grid_resolution=30)
with open("gambar/bab12/pdp.dat", "w") as f:
    f.write("recency p\n")
    for r, p in zip(pd_["grid_values"][0], pd_["average"][0]):
        f.write(f"{r:.1f} {p:.4f}\n")
print("PDP recency: p churn rata-rata naik dari "
      f"{pd_['average'][0][0]:.3f} ke {pd_['average'][0][-1]:.3f}")
