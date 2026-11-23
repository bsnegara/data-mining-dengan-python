import sys
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "kode/bab27")
from siapkan import DATA, NUMERIK, fitur  # noqa: E402
warnings.simplefilter("ignore")

lama = pd.read_csv(DATA + "pelanggan.csv")                       #<1>
X = fitur(lama, "2026-03-31")
d = pd.read_csv(DATA + "pelanggan_2026q3.csv")
d = d[d.kelompok == "kontrol"].reset_index(drop=True)
Xu = fitur(d, "2026-09-30")
Xl_num, Xu_num = X[NUMERIK], Xu[NUMERIK]
gbm = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.02,
                         num_leaves=4, reg_lambda=10, verbose=-1,
                         random_state=0).fit(Xl_num, lama.churn)
p_gbm = gbm.predict_proba(Xu_num)[:, 1]
print(f"LightGBM (atribut numerik) AUC uji: "
      f"{roc_auc_score(d.churn, p_gbm):.3f}")

nilai = shap.TreeExplainer(gbm).shap_values(Xu_num)              #<2>
nilai = nilai[1] if isinstance(nilai, list) else nilai
glob = pd.Series(np.abs(nilai).mean(0), NUMERIK).sort_values()[::-1]
print("\nrata-rata |SHAP| (log-odds), enam terbesar:")
print(glob[:6].round(3).to_string())
g8 = glob[:8][::-1].rename("shap")
g8.index = g8.index.str.replace("_", r"\_")
g8.to_csv("gambar/bab28/shap_global.dat", sep=" ",
          index_label="atribut")

model = joblib.load("kode/bab27/model/churn_v1.joblib")          #<3>
pi = permutation_importance(model, Xu, d.churn, scoring="roc_auc",
                            n_repeats=10, random_state=0)
pim = pd.Series(pi.importances_mean, Xu.columns).sort_values()[::-1]
print("\npermutation importance logistik (turunnya AUC):")
print(pim[:5].round(4).to_string())

i = int(np.argmax(p_gbm * (d.frekuensi_90h > 5)))                #<4>
dasar = shap.TreeExplainer(gbm).expected_value
dasar = float(np.ravel(dasar)[-1])
s = pd.Series(nilai[i], NUMERIK)
top = s.reindex(s.abs().sort_values()[::-1].index)[:5]
print(f"\nanggota {d.id_anggota[i]}: log-odds dasar {dasar:.2f}, "
      f"akhir {dasar + s.sum():.2f} "
      f"(p = {1 / (1 + np.exp(-(dasar + s.sum()))):.2f})")
for k, v in top.items():
    print(f"  {k:18s} = {Xu_num.loc[i, k]:>9.2f}  SHAP {v:+.2f}")
