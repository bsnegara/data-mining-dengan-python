import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "kode/bab27")
from siapkan import DATA, buat_model, fitur  # noqa: E402

model = joblib.load("kode/bab27/model/churn_v1.joblib")          #<1>
d = pd.read_csv(DATA + "pelanggan_2026q3.csv")
d = d[d.kelompok == "kontrol"].reset_index(drop=True)
d["p"] = model.predict_proba(fitur(d, "2026-09-30"))[:, 1]
d["sasar"] = d.p >= 0.278


def ringkas(df, grup):                                           #<2>
    baris = {}
    for g, x in df.groupby(grup):
        y, s = x.churn == 1, x.sasar
        baris[g] = {"churn": y.mean(), "skor": x.p.mean(),
                    "disasar": s.mean(), "TPR": s[y].mean(),
                    "FPR": s[~y].mean(), "presisi": y[s].mean()}
    hasil = pd.DataFrame(baris).T
    hasil.insert(0, "n", df.groupby(grup).size())
    return hasil


pd.set_option("display.float_format", "{:.3f}".format)  # noqa
print(ringkas(d, "jenis_kelamin").to_string())
print()
print(ringkas(d, "kota").sort_values("churn").to_string())

lama = pd.read_csv(DATA + "pelanggan.csv")                       #<3>
kat = ["kota", "tingkat"]
tanpa = buat_model(kategorik=kat).fit(
    fitur(lama, "2026-03-31", kat), lama.churn)
p2 = tanpa.predict_proba(fitur(d, "2026-09-30", kat))[:, 1]
auc = [roc_auc_score(d.churn, q) for q in (d.p, p2)]
print(f"\nAUC dengan jenis kelamin {auc[0]:.4f}, tanpa {auc[1]:.4f}")
print(f"korelasi skor: {np.corrcoef(d.p, p2)[0, 1]:.4f}; keputusan "
      f"berbeda: {((p2 >= 0.278) != d.sasar).sum()} anggota")
