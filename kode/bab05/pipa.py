import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "kode/data/"
df = pd.read_csv(DATA + "pelanggan_bersih.csv")
numerik = ["umur", "frekuensi_90h", "recency_hari", "nilai_90h",
           "porsi_promo", "porsi_aplikasi", "keluhan_90h", "jarak_km"]
kategorik = ["kota", "jenis_kelamin"]

praproses = ColumnTransformer([                                  #<1>
    ("num", Pipeline([("isi", SimpleImputer(strategy="median",
                                            add_indicator=True)),
                      ("skala", StandardScaler())]), numerik),
    ("kat", Pipeline([("isi", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]),
     kategorik)])
pipa = Pipeline([("praproses", praproses),                       #<2>
                 ("model", LogisticRegression(max_iter=1000,
                                              class_weight="balanced"))])

skor = cross_val_score(pipa, df[numerik + kategorik], df["churn"],
                       cv=5, scoring="roc_auc")                  #<3>
print("AUC per lipatan:", skor.round(3))
print("rata-rata      :", skor.mean().round(3))
pipa.fit(df[numerik + kategorik], df["churn"])
print(pipa.named_steps["praproses"].transform(
    df[numerik + kategorik]).shape)                              #<4>
