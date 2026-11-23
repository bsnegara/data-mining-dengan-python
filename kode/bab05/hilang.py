import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

DATA = "kode/data/"
df = pd.read_csv(DATA + "pelanggan_bersih.csv")
benar = pd.read_csv(DATA + "pelanggan.csv").set_index("id_anggota")["umur"]

kosong = df["umur"].isna()
print("umur kosong:", int(kosong.sum()))
print(df.groupby(kosong)[["churn", "frekuensi_90h"]]             #<1>
      .mean().round(3).rename(index={False: "terisi", True: "kosong"})
      .to_string())

def galat(isian):                                                #<2>
    t = benar.loc[df.loc[kosong, "id_anggota"]].values
    return np.sqrt(np.mean((isian[kosong.values] - t) ** 2))

rata = df["umur"].fillna(df["umur"].mean()).values
median = df["umur"].fillna(df["umur"].median()).values
kelompok = (df["umur"].fillna(                                   #<3>
    df.groupby("kota")["umur"]
    .transform("median"))).values
kol = ["umur", "porsi_pagi", "porsi_akhir_pekan", "porsi_kopi_hitam",
       "rata_nilai_struk", "porsi_promo"]
X = df[kol].fillna({"porsi_aplikasi": 0})
knn = KNNImputer(n_neighbors=15).fit_transform(                  #<4>
    (X - X.mean()) / X.std())[:, 0] * X["umur"].std() + X["umur"].mean()

for nama, isian in [("rata-rata", rata), ("median", median),
                    ("median per kota", kelompok), ("kNN", knn)]:
    print(f"{nama:<16} RMSE = {galat(isian):.2f} tahun")
penanda = SimpleImputer(strategy="median", add_indicator=True)  #<5>
print(penanda.fit_transform(df[["umur"]]).shape)
