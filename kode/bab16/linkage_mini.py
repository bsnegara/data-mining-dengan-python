import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

nama = ["G1", "G2", "G3", "G4", "G5", "G6"]
X = np.array([[0.85, 0.80], [0.90, 0.85], [0.85, 1.00],         #<1>
              [1.15, 1.00], [1.10, 1.05], [1.20, 0.95]])
D = squareform(pdist(X))                                         #<2>
print("matriks jarak:")
print("     " + "".join(f"{n:>6s}" for n in nama))
for n, r in zip(nama, D):
    print(f"{n:4s} " + "".join(f"{v:6.3f}" for v in r))
for metode in ["single", "complete"]:                            #<3>
    Z = linkage(X, metode)
    print(f"\n{metode} linkage:")
    kl = {i: [nama[i]] for i in range(6)}
    for t, (a, b, h, n) in enumerate(Z):
        kl[6 + t] = kl[int(a)] + kl[int(b)]
        print(f"  langkah {t + 1}: gabung {{{','.join(kl[int(a)])}}} + "
              f"{{{','.join(kl[int(b)])}}} pada jarak {h:.3f}")
