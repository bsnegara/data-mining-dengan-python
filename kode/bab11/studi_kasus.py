import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   StandardScaler)

DATA = "kode/data/"
df = pd.read_csv(DATA + "pelanggan.csv")
df["tren"] = df["frekuensi_90h"] / (df["frekuensi_90h_lalu"] + 1)  #<1>
miring = ["recency_hari", "frekuensi_90h", "nilai_90h", "jarak_km"]
lain = ["frekuensi_90h_lalu", "rata_nilai_struk", "porsi_promo",
        "porsi_aplikasi", "porsi_kopi_hitam", "porsi_pagi",
        "porsi_akhir_pekan", "ragam_menu", "keluhan_90h",
        "pakai_ulasan", "umur", "tren"]
kat = ["tingkat", "kota", "jenis_kelamin"]


def praproses(log):                                              #<2>
    m = (make_pipeline(FunctionTransformer(np.log1p,
                                           feature_names_out="one-to-one"),
                       StandardScaler()) if log else StandardScaler())
    return ColumnTransformer([("miring", m, miring),
                              ("num", StandardScaler(), lain),
                              ("kat", OneHotEncoder(), kat)])


Xl, Xu, yl, yu = train_test_split(df, df["churn"], test_size=0.3,
                                  stratify=df["churn"], random_state=7)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for log in [False, True]:
    s = cross_val_score(make_pipeline(praproses(log), LogisticRegression(
        max_iter=3000)), Xl, yl, cv=cv, scoring="roc_auc").mean()
    print(f"log1p pada atribut menceng = {log!s:5s}: AUC CV {s:.4f}")

model = make_pipeline(praproses(False), LogisticRegressionCV(    #<3>
    Cs=np.logspace(-3, 2, 11), cv=cv, scoring="neg_log_loss",
    max_iter=3000))
model.fit(Xl, yl)
lr = model[-1]
p = model.predict_proba(Xu)[:, 1]
print(f"C terpilih {lr.C_[0]:.3f}; uji: AUC {roc_auc_score(yu, p):.3f}, "
      f"Brier {brier_score_loss(yu, p):.4f}")

nama = [n.split("__")[1] for n in model[0].get_feature_names_out()]
koef = pd.Series(lr.coef_[0], index=nama)
print("\nkoefisien terbesar (per simpangan baku):")
print(koef[koef.abs().sort_values(ascending=False).index[:8]].round(2)
      .to_string())

skor = model.predict_proba(df)[:, 1]                             #<4>
sasaran = skor >= 15 / 54
print(f"\nskor >= 0.278: {sasaran.mean():.1%} anggota "
      f"(~{sasaran.mean() * 26000:.0f} dari 26.000)")
laba = 0.3 * 180_000 * skor[sasaran].sum() - 15_000 * sasaran.sum()  #<5>
print(f"harapan laba: Rp{laba / 1e6:.1f} jt per 8.000 anggota, "
      f"Rp{laba / 1e6 * 26000 / 8000:.1f} jt per 26.000")
print("sasaran per tingkat:", df[sasaran].groupby("tingkat").size()
      .to_dict())

Z = model[0].transform(df)                                       #<6>
i = int(np.argsort(-skor)[500])
sumbang = pd.Series(Z[i] * lr.coef_[0], index=nama)
print(f"\nanggota {df['id_anggota'].iloc[i]}, skor {skor[i]:.2f}; "
      "tiga alasan teratas:")
print(sumbang.sort_values(ascending=False).head(3).round(2).to_string())
print(df.loc[i, ["recency_hari", "tren", "keluhan_90h", "tingkat"]]
      .to_dict())
