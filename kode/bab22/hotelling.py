import numpy as np
import pandas as pd
from scipy import stats

b = pd.read_csv("kode/data/batch_sangrai.csv")
KOL = ["berat_kg", "durasi_detik", "suhu_akhir", "detik_crack"]
X = b[KOL].to_numpy(float)
f1 = X[:30]                                                      #<1>
m, p = len(f1), len(KOL)
mu = f1.mean(0)
S = np.cov(f1, rowvar=False)
Sinv = np.linalg.inv(S)
d = X - mu
T2 = np.einsum("ij,jk,ik->i", d, Sinv, d)                        #<2>
bka = p * (m + 1) * (m - 1) / (m * (m - p)) \
    * stats.f.ppf(0.9973, p, m - p)                              #<3>
print(f"batas kendali T2 (fase II, alfa 0,0027): {bka:.2f}")
urut = np.argsort(-T2)[:6]
print("enam T2 terbesar:")
for i in urut:
    z = d[i] / np.sqrt(np.diag(S))                               #<4>
    j = int(np.argmax(np.abs(z)))
    tanda = "SINYAL" if T2[i] > bka else "      "
    print(f"  {b.id_batch[i]} T2 {T2[i]:5.1f} {tanda} terbesar:"
          f" {KOL[j]} (z {z[j]:+.1f})")

r = np.corrcoef(f1, rowvar=False)
print(f"korelasi durasi-crack fase I: {r[1, 3]:.2f}")
with open("gambar/bab22/t2.dat", "w") as f:
    f.write("i t2\n")
    for i, v in enumerate(T2, 1):
        f.write(f"{i} {v:.2f}\n")
