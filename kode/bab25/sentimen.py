import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from normalisasi import bersihkan

d = pd.read_csv("kode/data/ulasan.csv", parse_dates=["tanggal"])  #<1>
latih = d.tanggal < "2026-02-01"
print(f"latih {latih.sum()} ulasan, uji {(~latih).sum()} ulasan")
print("sebaran sentimen:", d.sentimen.value_counts().to_dict())

varian = {"mentah": dict(stopword=False, stemming=False,         #<2>
                         negasi=False),
          "stop+stem": dict(negasi=False),
          "stop+stem+negasi": {}}
print("\nmakro-F1 data uji (unigram+bigram TF-IDF):")
print(f"{'praproses':18s} {'NB':>6s} {'logistik':>9s}")
for nama, opsi in varian.items():
    teks = d.teks.map(lambda t: bersihkan(t, **opsi))
    tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                         sublinear_tf=True)
    Xl, Xu = tv.fit_transform(teks[latih]), tv.transform(teks[~latih])
    f = []
    for m in (MultinomialNB(alpha=0.5),
              LogisticRegression(C=5, max_iter=2000,
                                 class_weight="balanced")):
        m.fit(Xl, d.sentimen[latih])
        f.append(f1_score(d.sentimen[~latih], m.predict(Xu),
                          average="macro"))
    print(f"{nama:18s} {f[0]:6.3f} {f[1]:9.3f}   "
          f"({Xl.shape[1]} atribut)")

pred = m.predict(Xu)                                             #<3>
print("\nlogistik, praproses lengkap:")
print(classification_report(d.sentimen[~latih], pred, digits=3))

kata = tv.get_feature_names_out()                                #<4>
for kelas in ("negatif", "positif"):
    k = list(m.classes_).index(kelas)
    top = np.argsort(m.coef_[k])[::-1][:6]
    print(f"{kelas}: " + ", ".join(kata[top]))

uji = d[~latih].assign(tebakan=pred)                             #<5>
salah = uji[uji.tebakan != uji.sentimen]
print(f"\n{len(salah)} salah tebak, contoh:")
for _, r in salah.sample(4, random_state=3).iterrows():
    print(f"  [{r.rating}] {r.teks[:60]}")
    print(f"      label {r.sentimen}, tebakan {r.tebakan}")
