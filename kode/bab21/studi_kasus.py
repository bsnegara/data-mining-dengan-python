import sys; sys.path.insert(0, "kode/bab21")
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from siapkan import matriks_panel

D, nama = matriks_panel()                                        #<1>
iu = np.triu_indices(len(D), 1)
def nmds(D, seed=0):
    m = MDS(2, metric_mds=False, metric="precomputed", n_init=16,
            init="random", random_state=seed, normalized_stress=False)
    return m.fit_transform(D)
X = nmds(D)                                                      #<2>
d = pdist(X)
dhat = IsotonicRegression().fit_transform(D[iu], d)
stress = np.sqrt(((d - dhat) ** 2).sum() / (d ** 2).sum())
rho = spearmanr(D[iu], d).statistic
print(f"MDS nonmetrik 2D: stress-1 {stress:.3f}")
print(f"Spearman(skor panel, jarak peta) {rho:.3f}")

rasa = pd.read_csv("kode/data/profil_rasa.csv").set_index("kopi")   #<3>
rasa = rasa.loc[nama]
A = StandardScaler().fit_transform(rasa.iloc[:, 2:])
P = PCA(2).fit_transform(A)
_, _, beda = procrustes(X, P)
print(f"Procrustes peta panel vs peta PCA atribut: {1 - beda:.3f}")

R, _ = orthogonal_procrustes(X - X.mean(0), P - P.mean(0))     #<4>
X = (X - X.mean(0)) @ R                  # putar agar sumbu 1 ~ PC1
ATR = ["tanah", "pahit", "keasaman", "bunga", "manis",
       "cokelat", "rasa_ikutan"]
Xs = X / X.std(0)
print("\npencocokan atribut ke peta (arah dan R^2):")
panah = {}
for a in ATR:
    lr = LinearRegression().fit(Xs, rasa[a])
    arah = lr.coef_ / np.linalg.norm(lr.coef_)
    panah[a] = arah * np.sqrt(lr.score(Xs, rasa[a]))
    print(f"  {a:11s} arah ({arah[0]:+.2f}, {arah[1]:+.2f})"
          f"  R^2 {lr.score(Xs, rasa[a]):.2f}")

print("\nkesesuaian setiap panelis dengan peta bersama:")        #<5>
for p in ["made", "ayu", "rizal", "tiur", "yosef"]:
    Dp, _ = matriks_panel(f"skor_{p}")
    _, _, b = procrustes(X, nmds(Dp))
    r = spearmanr(Dp[iu], D[iu]).statistic
    print(f"  {p:6s} Spearman dgn rata-rata {r:.2f}, "
          f"Procrustes {1 - b:.3f}")

stabil = []                                                      #<6>
for s in range(1, 6):
    _, _, b = procrustes(X, nmds(D, seed=s))
    stabil.append(1 - b)
print(f"\nkestabilan antar-inisialisasi: {min(stabil):.3f}"
      f" - {max(stabil):.3f}")

with open("gambar/bab21/peta.dat", "w") as f:
    f.write("x y pulau nama\n")
    kode = {p: i for i, p in enumerate(rasa.pulau.unique())}
    for k, (x, y) in zip(nama, Xs):
        f.write(f"{x:.3f} {y:.3f} {kode[rasa.pulau[k]]} {{{k}}}\n")
with open("gambar/bab21/panah.dat", "w") as f:
    for a, v in panah.items():
        f.write(f"{v[0]:.3f} {v[1]:.3f} {{{a.replace('_', ' ')}}}\n")
with open("gambar/bab21/shepard.dat", "w") as f:
    f.write("skor jarak disparitas\n")
    o = np.argsort(D[iu])
    for i in o:
        f.write(f"{D[iu][i]:.2f} {d[i]:.3f} {dhat[i]:.3f}\n")
print("urutan pulau:", list(kode))
