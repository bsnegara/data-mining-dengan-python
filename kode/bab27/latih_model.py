import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from siapkan import DATA, buat_model, fitur

d = pd.read_csv(DATA + "pelanggan.csv")                          #<1>
X, y = fitur(d, "2026-03-31"), d.churn
print(f"data latih: {len(d)} anggota per 31 Maret 2026, "
      f"churn {y.mean():.3f}")

cv = StratifiedKFold(5, shuffle=True, random_state=0)            #<2>
p = cross_val_predict(buat_model(), X, y, cv=cv,
                      method="predict_proba")[:, 1]
print(f"AUC-CV {roc_auc_score(y, p):.3f}, Brier "
      f"{brier_score_loss(y, p):.4f}")
desil = pd.qcut(p, 10, labels=False)
kal = pd.DataFrame({"p": p, "y": y}).groupby(desil).mean()
print("kalibrasi per desil (rata-rata tebakan vs nyata):")
print("  " + "  ".join(f"{a:.2f}/{b:.2f}" for a, b in
                       kal.to_numpy()[[0, 3, 6, 8, 9]]))

model = buat_model().fit(X, y)                                   #<3>
folder = Path("kode/bab27/model")
folder.mkdir(exist_ok=True)
joblib.dump(model, folder / "churn_v1.joblib")
meta = {"versi": "churn-v1", "dilatih": "2026-04-07",
        "data": "pelanggan.csv (potong 2026-03-31)",
        "auc_cv": round(roc_auc_score(y, p), 4),
        "sklearn": sklearn.__version__,
        "atribut": list(X.columns),
        "skor_acuan": np.quantile(p, np.linspace(0, 1, 11)).tolist()}
(folder / "churn_v1.json").write_text(json.dumps(meta, indent=1))
print("tersimpan:", sorted(f.name for f in folder.iterdir()))

koef = pd.Series(model[-1].coef_[0],                             #<4>
                 index=model[0].get_feature_names_out())
num = koef[koef.index.str.startswith("num__")]
num.index = num.index.str[5:]
print("\nkoefisien atribut numerik terbesar (dibakukan):")
for k in num.abs().sort_values(ascending=False).index[:6]:
    print(f"  {k:22s} {num[k]:+.2f}")
