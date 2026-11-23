import sys; sys.path.insert(0, "kode/bab21")
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.isotonic import IsotonicRegression
from sklearn.manifold import MDS
from siapkan import matriks_panel

D, nama = matriks_panel()                                        #<1>
iu = np.triu_indices(len(D), 1)

def stress1(D, X):                                               #<2>
    """Stress-1 Kruskal: jarak peta vs disparitas monoton."""
    d = pdist(X)
    dhat = IsotonicRegression().fit_transform(D[iu], d)
    return np.sqrt(((d - dhat) ** 2).sum() / (d ** 2).sum()), d, dhat

print("dim  stress metrik  stress nonmetrik")
hasil = []
for k in range(1, 6):                                            #<3>
    s = {}
    for metrik in [True, False]:
        mds = MDS(k, metric_mds=metrik, metric="precomputed",
                  n_init=8, init="random", random_state=0,
                  normalized_stress=False)
        X = mds.fit_transform(D)
        s[metrik] = stress1(D, X)[0]
    hasil.append((k, s[True], s[False]))
    print(f"{k:3d}  {s[True]:13.3f}  {s[False]:16.3f}")
with open("gambar/bab21/stress.dat", "w") as f:
    f.write("k metrik nonmetrik\n")
    for k, a, b in hasil:
        f.write(f"{k} {a:.4f} {b:.4f}\n")
