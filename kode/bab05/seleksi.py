import pandas as pd
from sklearn.feature_selection import (RFE, VarianceThreshold,
                                       mutual_info_classif)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA = "kode/data/"
df = pd.read_csv(DATA + "pelanggan_bersih.csv").dropna()
df["tren_frek"] = df["frekuensi_90h"] / (df["frekuensi_90h_lalu"] + 1)
fitur = ["umur", "frekuensi_90h", "tren_frek", "recency_hari",
         "nilai_90h", "rata_nilai_struk", "porsi_promo",
         "porsi_aplikasi", "porsi_kopi_hitam", "porsi_pagi",
         "porsi_akhir_pekan", "ragam_menu", "keluhan_90h",
         "jarak_km", "pakai_ulasan"]
X, y = df[fitur], df["churn"]

vt = VarianceThreshold(0.01).fit(X)                              #<1>
print("varians kecil:", list(X.columns[~vt.get_support()]))

mi = mutual_info_classif(X, y, random_state=0)                   #<2>
mi = pd.Series(mi, index=fitur).sort_values(ascending=False)
print(mi.round(3).head(6).to_string())

Xs = StandardScaler().fit_transform(X)
rfe = RFE(LogisticRegression(max_iter=1000),                     #<3>
          n_features_to_select=6).fit(Xs, y)
import textwrap
print(textwrap.fill("RFE: " + ", ".join(X.columns[rfe.support_]), 58))
