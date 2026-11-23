import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfVectorizer)
from normalisasi import bersihkan

d = pd.read_csv("kode/data/ulasan.csv")
teks = d.teks.map(bersihkan)
buang = ["jujur", "overall", "coba", "kali", "top", "recommended",  #<1>
         "sukses", "terus", "balik", "pasti", "sejauh", "parah",
         "kecewa", "lumayan", "biasa", "tolong", "deh", "sih"]
label = d.topik.str.split(";")


def tampilkan(nama, model, kata, W):                             #<2>
    print(f"\n{nama}:")
    utama = W.argmax(1)
    for j, komp in enumerate(model.components_):
        top = ", ".join(kata[np.argsort(komp)[::-1][:6]])
        grup = label[utama == j]
        lab = grup.explode().value_counts() / len(grup)
        print(f"  {j}: {top}")
        print(f"     {len(grup)} ulasan; {lab.index[0]} di "
              f"{lab.iloc[0]:.0%}, {lab.index[1]} di "
              f"{lab.iloc[1]:.0%}")


cv = CountVectorizer(stop_words=buang, min_df=5)                 #<3>
Xc = cv.fit_transform(teks)
lda = LatentDirichletAllocation(6, learning_method="batch",
                                random_state=0, max_iter=50)
tampilkan("LDA", lda, cv.get_feature_names_out(),
          lda.fit_transform(Xc))

tv = TfidfVectorizer(stop_words=buang, min_df=5)                 #<4>
Xt = tv.fit_transform(teks)
nmf = NMF(6, init="nndsvda", random_state=0, max_iter=500)
tampilkan("NMF", nmf, tv.get_feature_names_out(),
          nmf.fit_transform(Xt))
