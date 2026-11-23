import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "kode/bab15")
from siapkan import PERILAKU, muat

df, X = muat()
Z = StandardScaler().fit_transform(X)
baris, kolom = 8, 8                                              #<1>
rng = np.random.default_rng(0)
W = rng.normal(0, 0.1, (baris * kolom, Z.shape[1]))
pos = np.array([(i, j) for i in range(baris) for j in range(kolom)])
T = 20 * len(Z)
for t in range(T):                                               #<2>
    x = Z[rng.integers(len(Z))]
    eta = 0.5 * (0.01 / 0.5) ** (t / T)
    sigma = 4.0 * (0.5 / 4.0) ** (t / T)
    j = ((W - x) ** 2).sum(1).argmin()                           #<3>
    h = np.exp(-((pos - pos[j]) ** 2).sum(1) / (2 * sigma ** 2))  #<4>
    W += eta * h[:, None] * (x - W)
bmu = ((Z[:, None, :] - W[None]) ** 2).sum(2).argmin(1)
qe = np.sqrt(((Z - W[bmu]) ** 2).sum(1)).mean()                  #<5>
print(f"galat kuantisasi rata-rata {qe:.3f}; neuron kosong "
      f"{(np.bincount(bmu, minlength=64) == 0).sum()}")
km = KMeans(5, n_init=20, random_state=0).fit(Z)                 #<6>
seg = {0: "mahasiswa", 1: "pekerja", 2: "keluarga", 3: "tidur",
       4: "penikmat"}
mayor = pd.Series(km.labels_).groupby(bmu).agg(lambda s: s.mode()[0])
umat = np.zeros(baris * kolom)                                   #<7>
for i, (a, b) in enumerate(pos):
    tetangga = [k for k, (c, e) in enumerate(pos)
                if abs(a - c) + abs(b - e) == 1]
    umat[i] = np.mean([np.linalg.norm(W[i] - W[k]) for k in tetangga])
with open("gambar/bab16/som.dat", "w") as f:
    f.write("i j n seg u\n")
    for i, (a, b) in enumerate(pos):
        f.write(f"{b} {a} {(bmu == i).sum()} {mayor.get(i, -1)} "
                f"{umat[i]:.3f}\n")
print("segmen dominan per baris peta (0-4, - = kosong):")
for a in range(baris):
    print("  " + " ".join(str(mayor.get(a * kolom + b, "-"))
                          for b in range(kolom)))
w = pd.DataFrame(W, columns=PERILAKU)
print("neuron dengan porsi kopi hitam tertinggi:",
      int(w["porsi_kopi_hitam"].idxmax()), "; terendah frekuensi:",
      int(w["frekuensi_90h"].idxmin()))
