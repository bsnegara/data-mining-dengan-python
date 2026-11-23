import sys; sys.path.insert(0, "kode/bab19")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from jb import JaringanBayes, muat_biji

d = muat_biji()
latih = d[d.tanggal < "2025-07-01"]                              #<1>
uji = d[d.tanggal >= "2025-07-01"]
print(f"latih {len(latih)} kiriman, uji {len(uji)} kiriman; "
      f"porsi cacat latih {np.mean(latih.cacat == 'ya'):.3f}")
NILAI = {v: sorted(d[v].unique()) for v in d.columns
         if v != "tanggal"}
AKAR = {"musim": [], "jemur": [], "basah": [], "simpan": []}
STRUKTUR = {                                                     #<2>
    "kosong": {v: [] for v in NILAI},
    "dipelajari": {**AKAR, "air": ["musim", "jemur", "basah"],
                   "jamur": ["air", "simpan"], "cacat": ["jamur"]},
    "pakar": {**AKAR, "air": ["musim", "jemur", "basah"],
              "jamur": ["air", "simpan"],
              "cacat": ["jamur", "jemur"]},
}
print("\nlog-likelihood rata-rata per kiriman uji:")
model = {}
for nama, induk in STRUKTUR.items():
    model[nama] = JaringanBayes(induk, NILAI).latih(latih)
    print(f"  {nama:10s} {model[nama].log_lik(uji) / len(uji):.3f}")

jb = model["pakar"]
gab = jb.gabungan()                                              #<3>
TERIMA = ["musim", "jemur", "basah", "simpan", "air"]
def p_cacat(baris):
    g = gab
    for v in TERIMA:
        g = g[g[v] == baris[v]]
    return g.p[g.cacat == "ya"].sum() / g.p.sum()
p_bn = uji.apply(p_cacat, axis=1)
X = pd.get_dummies(d[TERIMA], drop_first=True).astype(float)
lr = LogisticRegression(max_iter=1000)
lr.fit(X.loc[latih.index], latih.cacat == "ya")
p_lr = lr.predict_proba(X.loc[uji.index])[:, 1]
y = (uji.cacat == "ya").to_numpy()
print(f"\nramalan cacat saat penerimaan (uji, {y.sum()} cacat):")
for nama, p in [("jaringan Bayes", p_bn), ("regresi logistik", p_lr)]:
    print(f"  {nama:17s} AUC {roc_auc_score(y, p):.3f}"
          f"  Brier {brier_score_loss(y, p):.4f}")

print("\nP(cacat) di bawah intervensi:")                        #<4>
dasar = jb.tanya("cacat")["ya"]
SKENARIO = {"tanpa intervensi": {},
            "do(jemur=mesin)": {"jemur": "mesin"},
            "do(simpan=singkat)": {"simpan": "singkat"},
            "do(air=normal)": {"air": "normal"},
            "do(air=normal, simpan=singkat)":
                {"air": "normal", "simpan": "singkat"}}
hasil = {}
for nama, iv in SKENARIO.items():
    hasil[nama] = h = jb.tanya("cacat", intervensi=iv)["ya"]
    print(f"  {nama:31s} {h:.3f}  ({h - dasar:+.3f})")

print("\nmengamati versus mengintervensi kadar air:")            #<5>
for s in ["tinggi", "normal"]:
    amati = jb.tanya("cacat", bukti={"air": s})["ya"]
    ubah = jb.tanya("cacat", intervensi={"air": s})["ya"]
    print(f"  air={s:6s}  P(cacat | air) {amati:.3f}"
          f"   P(cacat | do(air)) {ubah:.3f}")

print("\ndiagnosis kiriman cacat: P(penyebab | cacat=ya)")      #<6>
for v, s in [("air", "tinggi"), ("jamur", "ya"), ("jemur", "terpal"),
             ("simpan", "lama"), ("musim", "hujan")]:
    pri = jb.tanya(v)[s]
    pos = jb.tanya(v, bukti={"cacat": "ya"})[s]
    print(f"  {v + '=' + s:13s} prior {pri:.3f}"
          f" -> posterior {pos:.3f}")

KIRIMAN, RUGI = 500, 6_000_000                                   #<7>
print(f"\nperkiraan kiriman cacat per tahun (dari {KIRIMAN}):")
for nama in ["tanpa intervensi", "do(air=normal)",
             "do(air=normal, simpan=singkat)"]:
    k = KIRIMAN * hasil[nama]
    print(f"  {nama:31s} {k:5.1f} kiriman,"
          f" rugi Rp{k * RUGI / 1e6:4.0f} jt")
