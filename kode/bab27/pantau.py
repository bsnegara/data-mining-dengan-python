import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from siapkan import DATA, NUMERIK, fitur

model = joblib.load("kode/bab27/model/churn_v1.joblib")          #<1>
meta = json.load(open("kode/bab27/model/churn_v1.json"))
lama = fitur(pd.read_csv(DATA + "pelanggan.csv"), "2026-03-31")
d = pd.read_csv(DATA + "pelanggan_2026q3.csv")
baru = fitur(d, "2026-09-30")


def psi(a, b, k=10):                                             #<2>
    """Population stability index: a = acuan, b = baru."""
    batas = np.unique(np.quantile(a, np.linspace(0, 1, k + 1)))
    batas[0], batas[-1] = -np.inf, np.inf
    pa = np.histogram(a, batas)[0] / len(a) + 1e-4
    pb = np.histogram(b, batas)[0] / len(b) + 1e-4
    return float(np.sum((pb - pa) * np.log(pb / pa)))


nilai = pd.Series({c: psi(lama[c], baru[c]) for c in NUMERIK})   #<3>
print("PSI atribut (> 0,25 = bergeser besar, 0,1-0,25 = waspada):")
for c, v in nilai.sort_values(ascending=False)[:5].items():
    print(f"  {c:18s} {v:.3f}  rata-rata {lama[c].mean():8.2f}"
          f" -> {baru[c].mean():8.2f}")

p_lama = model.predict_proba(lama)[:, 1]                         #<4>
p_baru = model.predict_proba(baru)[:, 1]
print(f"\nPSI skor churn: {psi(p_lama, p_baru):.3f}; rata-rata skor "
      f"{p_lama.mean():.3f} -> {p_baru.mean():.3f}")

k = d.kelompok == "kontrol"                                      #<5>
print(f"\nkinerja pada kelompok kontrol Q3 (n = {k.sum()}):")
print(f"  AUC {roc_auc_score(d.churn[k], p_baru[k]):.3f} "
      f"(saat dilatih {meta['auc_cv']:.3f})")
print(f"  churn nyata {d.churn[k].mean():.3f}, rata-rata tebakan "
      f"{p_baru[k].mean():.3f}")
print(f"  AUC bila kelompok voucer ikut dihitung: "
      f"{roc_auc_score(d.churn, p_baru):.3f}")
