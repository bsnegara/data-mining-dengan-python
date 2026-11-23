import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

DATA = "kode/data/"
df = pd.read_csv(DATA + "pelanggan_bersih.csv")

oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")  #<1>
kota = oh.fit_transform(df[["kota"]])
print(kota.shape, list(oh.get_feature_names_out())[:3])

urutan = [["perunggu", "perak", "emas"]]                         #<2>
od = OrdinalEncoder(categories=urutan)
print(od.fit_transform(df[["tingkat"]])[:5].ravel())

print(pd.get_dummies(df["jenis_kelamin"], prefix="jk",           #<3>
                     dummy_na=True).head(3).astype(int).to_string())

target = df.groupby("kota")["churn"].mean()                      #<4>
print(target.round(3).sort_values().to_string())
