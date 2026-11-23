import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV, lasso_path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DATA = "kode/data/"
df = pd.read_csv(DATA + "pelanggan.csv")
fitur = ["umur", "frekuensi_90h", "frekuensi_90h_lalu", "recency_hari",
         "nilai_90h", "rata_nilai_struk", "porsi_promo", "porsi_aplikasi",
         "porsi_kopi_hitam", "porsi_pagi", "porsi_akhir_pekan",
         "ragam_menu", "keluhan_90h", "jarak_km", "pakai_ulasan"]
gerai = df[df["gerai_utama"] == "G24"]                           #<1>
X, y = gerai[fitur], gerai["belanja_bulan_depan"] / 1000
print(f"{len(gerai)} anggota gerai G24, 15 atribut")

cv = KFold(10, shuffle=True, random_state=0)
poli = PolynomialFeatures(2, include_bias=False)                 #<2>
model = {
    "OLS, 15 atribut": make_pipeline(StandardScaler(),
                                     LinearRegression()),
    "OLS, 135 atribut": make_pipeline(poli, StandardScaler(),
                                      LinearRegression()),
    "ridge, 135 atribut": make_pipeline(poli, StandardScaler(),
                                        RidgeCV(np.logspace(-2, 4, 60))),
    "lasso, 135 atribut": make_pipeline(poli, StandardScaler(),
                                        LassoCV(cv=5, max_iter=50000))}
for n, m in model.items():
    mae = -cross_val_score(m, X, y, cv=cv,
                           scoring="neg_mean_absolute_error").mean()
    print(f"{n:20s} MAE CV {mae:7.1f} ribu Rp")

m = model["lasso, 135 atribut"].fit(X, y)                        #<3>
koef = m[-1].coef_
nama = m[0].get_feature_names_out(fitur)
print(f"lasso memilih {(koef != 0).sum()} dari 135 atribut; terbesar:")
for i in np.argsort(-np.abs(koef))[:5]:
    print(f"  {nama[i]:30s} {koef[i]:+7.1f}")

Xs = StandardScaler().fit_transform(X)                           #<4>
alfa, jalur, _ = lasso_path(Xs, y - y.mean(), alphas=np.logspace(0.5, 2.6, 60))
pd.DataFrame(np.c_[np.log10(alfa), jalur.T], columns=["la"] + fitur
             ).round(3).to_csv("gambar/bab10/lasso.dat", sep=" ", index=False)
