import numpy as np
import pandas as pd

DATA = "kode/data/"
pel = pd.read_csv(DATA + "pelanggan.csv")
pel["nilai_90h"] = pel["nilai_90h"] / 1000
kol = ["umur", "frekuensi_90h", "recency_hari", "nilai_90h"]

bentuk = pd.DataFrame({  # nilai_90h dalam ribu rupiah
    "rata": pel[kol].mean(),
    "median": pel[kol].median(),
    "kemiringan": pel[kol].skew(),                                #<1>
    "keruncingan": pel[kol].kurt(),                               #<2>
})
print(bentuk.round(2).to_string())

x = pel["nilai_90h"]
q1, q3 = x.quantile([.25, .75])                                   #<3>
iqr = q3 - q1
batas = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
print("Q1, Q3, IQR :", q1, q3, iqr)
print("di luar pagar:", int(((x < batas[0]) | (x > batas[1])).sum()))

positif = x[x > 0]                                                #<4>
print("nilai nol:", int((x == 0).sum()))
print("kemiringan log10:", round(np.log10(positif).skew(), 2))
