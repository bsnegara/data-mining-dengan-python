import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

a = pd.read_csv("kode/data/audit_kasir.csv", parse_dates=["tanggal"])
a["rasio_batal"] = a.struk_batal / a.jumlah_struk                 #<1>
a["diskon_per_struk"] = a.diskon_manual / a.jumlah_struk
FITUR = ["rasio_batal", "diskon_per_struk", "porsi_tunai",
         "struk_setelah_tutup"]
X = RobustScaler().fit_transform(a[FITUR])                       #<2>

CURANG = {"G07-K2": "2026-02-01", "G31-K3": "2026-02-01",        #<3>
          "G18-K1": "2026-01-20"}
y = np.array([k in CURANG and t >= pd.Timestamp(CURANG[k])
              for k, t in zip(a.id_kasir, a.tanggal)])
print(f"{len(a)} kasir-hari, {y.sum()} di antaranya curang "
      f"({y.mean():.1%})")

skor = {                                                         #<4>
    "z robust (maks)": np.abs(X).max(1),
    "LOF": -LocalOutlierFactor(n_neighbors=50).fit(X)
            .negative_outlier_factor_,
    "Isolation Forest": -IsolationForest(n_estimators=300,
                                         random_state=0).fit(X)
                         .score_samples(X),
}
print("\ntingkat kasir-hari:    AUC    AP")                       #<5>
for nama, s in skor.items():
    print(f"  {nama:17s} {roc_auc_score(y, s):.3f}  "
          f"{average_precision_score(y, s):.3f}")

a["skor"] = skor["Isolation Forest"]                             #<6>
a["minggu"] = a.tanggal.dt.to_period("W-SUN")
mingguan = a.groupby(["minggu", "id_kasir"]).skor.mean().reset_index()
print("\ndaftar audit mingguan: 10 kasir dengan skor rata-rata")
print("tertinggi; berapa minggu kasir curang masuk daftar?")
for k, t in CURANG.items():
    mg = mingguan[mingguan.minggu.dt.end_time >= pd.Timestamp(t)]
    masuk = sum(k in g.nlargest(10, "skor").id_kasir.tolist()
                for _, g in mg.groupby("minggu"))
    print(f"  {k}: {masuk} dari {mg.minggu.nunique()} minggu")
daftar = [k for _, g in mingguan.groupby("minggu")
          for k in g.nlargest(10, "skor").id_kasir if k not in CURANG]
hit = pd.Series(daftar).value_counts()
print(f"kasir jujur yang pernah masuk daftar: {hit.size} dari 117")
print(f"paling sering: {hit.index[0]} ({hit.iloc[0]} minggu)")

feb = a[a.tanggal >= "2026-02-01"]                               #<7>
prof = feb.groupby("id_kasir")[FITUR].mean()
ref = prof.median()
print("\nprofil kasir yang ditandai vs median semua kasir (Feb-Mar):")
print(pd.concat([prof.loc[list(CURANG)], ref.rename("median")
                 .to_frame().T]).round(3).to_string())
