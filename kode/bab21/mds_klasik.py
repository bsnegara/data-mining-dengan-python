import numpy as np
import pandas as pd

def mds_klasik(D, k=2):                                          #<1>
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    nilai, vektor = np.linalg.eigh(B)
    urut = np.argsort(nilai)[::-1]
    nilai, vektor = nilai[urut], vektor[:, urut]
    return vektor[:, :k] * np.sqrt(np.maximum(nilai[:k], 0)), nilai

D = np.array([[0, 3, 4], [3, 0, 5], [4, 5, 0]], float)           #<2>
X, nilai = mds_klasik(D)
print("segitiga 3-4-5, nilai eigen B:", nilai.round(3))
print("koordinat:\n", X.round(3))

KOTA = {"Banda Aceh": (5.55, 95.32), "Medan": (3.59, 98.67),     #<3>
        "Pekanbaru": (0.51, 101.45), "Padang": (-0.95, 100.35),
        "Jambi": (-1.61, 103.61), "Bengkulu": (-3.80, 102.27),
        "Palembang": (-2.98, 104.75),
        "Bandar Lampung": (-5.43, 105.26)}
lat, lon = np.radians(list(KOTA.values())).T
dlat = lat[:, None] - lat[None]
dlon = lon[:, None] - lon[None]
h = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None]) \
    * np.sin(dlon / 2) ** 2
D = 2 * 6371 * np.arcsin(np.sqrt(h))                             # km
print("\njarak udara (km), sebagian:")
print(pd.DataFrame(D, KOTA, KOTA).iloc[:4, :4].round(0).to_string())
X, nilai = mds_klasik(D)                                         #<4>
print("nilai eigen terbesar:", (nilai[:4] / 1e3).round(1), "(ribu)")
peta = pd.DataFrame(X, KOTA, ["dim1", "dim2"]).round(0)
print(peta.to_string())
with open("gambar/bab21/kota.dat", "w") as f:
    f.write("x y nama\n")
    for k, r in peta.iterrows():
        f.write(f"{r.dim1:.0f} {r.dim2:.0f} {{{k}}}\n")
