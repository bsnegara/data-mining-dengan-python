import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from normalisasi import bersihkan

d = pd.read_csv("kode/data/ulasan.csv", parse_dates=["tanggal"])
d["bersih"] = d.teks.map(bersihkan)
latih = (d.tanggal < "2026-02-01").to_numpy()
mlb = MultiLabelBinarizer()                                      #<1>
Y = mlb.fit_transform(d.topik.str.split(";"))
TOPIK = list(mlb.classes_)

KUNCI = {"antar": ["kirim", "tumpah", "cair", "kemas"],          #<2>
         "aplikasi": ["aplikasi", "login", "error", "poin"],
         "harga": ["harga", "mahal", "murah", "promo"],
         "pelayanan": ["layan", "barista", "kasir", "antri",
                       "pegawai", "mbak"],
         "rasa": ["kopi", "rasa", "manis", "pahit", "espresso"],
         "suasana": ["tempat", "wifi", "acnya", "meja", "colok",
                     "berisik"]}
Yk = np.array([[any(k in t for k in KUNCI[tp]) for tp in TOPIK]
               for t in d.bersih]).astype(int)

tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xl, Xu = tv.fit_transform(d.bersih[latih]), tv.transform(
    d.bersih[~latih])
ovr = OneVsRestClassifier(LogisticRegression(C=10, max_iter=2000))
Yp = ovr.fit(Xl, Y[latih]).predict(Xu)                           #<3>
Yu, Yku = Y[~latih], Yk[~latih]
print("F1 per topik (uji):   kata kunci   logistik")
for j, tp in enumerate(TOPIK):
    print(f"  {tp:18s} {f1_score(Yu[:, j], Yku[:, j]):.3f}"
          f"      {f1_score(Yu[:, j], Yp[:, j]):.3f}")
mikro = [f1_score(Yu, y, average="micro") for y in (Yku, Yp)]
tepat = [(Yu == y).all(1).mean() for y in (Yku, Yp)]
print(f"  {'mikro-F1':18s} {mikro[0]:.3f}      {mikro[1]:.3f}")
print(f"  {'semua tepat':18s} {tepat[0]:.3f}      {tepat[1]:.3f}")

neg = (d.sentimen == "negatif").to_numpy()                       #<4>
clf = LogisticRegression(C=5, max_iter=2000, class_weight="balanced")
p_cv = cross_val_predict(clf, Xl, neg[latih], cv=5,
                         method="predict_proba")[:, 1]
ambang = max(a for a in np.arange(.05, .95, .01)
             if (p_cv[neg[latih]] >= a).mean() >= .95)
pu = clf.fit(Xl, neg[latih]).predict_proba(Xu)[:, 1] >= ambang
print(f"\nambang keluhan (recall CV >= 0,95): {ambang:.2f}")
print(f"uji: recall {recall_score(neg[~latih], pu):.3f}, presisi "
      f"{precision_score(neg[~latih], pu):.3f}, ditandai "
      f"{pu.mean():.1%} ulasan")

Xs = tv.transform(d.bersih)                                      #<5>
d["keluhan"] = clf.predict_proba(Xs)[:, 1] >= ambang
T = pd.DataFrame(ovr.predict(Xs), columns=TOPIK, index=d.index)
bulan = d.tanggal.dt.to_period("M")
tren = T[d.keluhan].groupby(bulan[d.keluhan]).sum()
tren = tren.div(d.groupby(bulan).size(), axis=0) * 100
print("\nkeluhan per topik per 100 ulasan:")
print(tren.round(1).to_string())
benar = pd.DataFrame(Y, columns=TOPIK)[neg].groupby(bulan[neg]).sum()
benar = benar.div(d.groupby(bulan).size(), axis=0) * 100
r = np.corrcoef(tren.to_numpy().ravel(), benar.to_numpy().ravel())
print(f"korelasi dengan hitungan label manual: {r[0, 1]:.3f}")
n = d.groupby(bulan).size()                                      #<6>
p0 = tren.mean() / 100
se = np.sqrt(np.outer(1 / n, p0 * (1 - p0)))
bka = 100 * (p0.to_numpy() + 3 * se)
sinyal = tren > bka
print("di atas batas kendali p:",
      [(str(b), t) for b, t in sinyal.stack()[sinyal.stack()].index]
      or "tidak ada")

baru = ["kurirnya nyasar, kopi dateng telat dan dingin",          #<7>
        "baristanya cuek bgt, dipanggil diem aja",
        "aplikasinya ngelag pas checkout",
        "harganya bikin dompet nangis",
        "kopinya enak, tempatnya gerah",
        "ga worth it, mending beli sachet"]
Pb = ovr.predict_proba(tv.transform([bersihkan(t) for t in baru]))
print("\nulasan dengan kata-kata baru:")
for t, p in zip(baru, Pb):
    top = np.argsort(p)[::-1][:2]
    tebak = ", ".join(f"{TOPIK[j]} {p[j]:.2f}" for j in top)
    print(f"  {t[:38]:38s} {tebak}")
tren.assign(t=range(1, len(tren) + 1)).to_csv(
    "gambar/bab25/tren.dat", sep=" ", index=False,
    float_format="%.2f")
