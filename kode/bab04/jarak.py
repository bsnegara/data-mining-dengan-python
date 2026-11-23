import numpy as np
import pandas as pd
from scipy.spatial import distance

DATA = "kode/data/"
pel = pd.read_csv(DATA + "pelanggan.csv", index_col="id_anggota")
kol = ["frekuensi_90h", "recency_hari", "nilai_90h"]
tiga = pel.loc[["A00001", "A00002", "A00003"], kol]
print(tiga.to_string())

a, b = tiga.values[0], tiga.values[1]
print("Euclides :", round(distance.euclidean(a, b), 1))          #<1>
print("Manhattan:", round(distance.cityblock(a, b), 1))
print("Chebyshev:", round(distance.chebyshev(a, b), 1))

z = (pel[kol] - pel[kol].mean()) / pel[kol].std()                 #<2>
za, zb = z.loc["A00001"].values, z.loc["A00002"].values
print("Euclides baku:", round(distance.euclidean(za, zb), 2))
print("kosinus:", round(1 - distance.cosine(za, zb), 2))          #<3>

menu_a = {"K01", "M01", "H03", "N02"}                             #<4>
menu_b = {"K01", "M01", "K04"}
j = len(menu_a & menu_b) / len(menu_a | menu_b)
print("Jaccard:", round(j, 2))
