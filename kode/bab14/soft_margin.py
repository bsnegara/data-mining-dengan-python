import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from siapkan import muat

df, _, y = muat()
sampel = df.sample(150, random_state=3)                          #<1>
X = StandardScaler().fit_transform(
    sampel[["ketinggian_mdpl", "cacat_primer"]])
ys = np.where(sampel["specialty"] == 1, 1, -1)
for C in [0.01, 1, 100]:
    s = SVC(kernel="linear", C=C).fit(X, ys)
    w, b = s.coef_[0], s.intercept_[0]
    xi = np.maximum(0, 1 - ys * (X @ w + b))                     #<2>
    print(f"C={C:<5g} w=({w[0]:+.2f},{w[1]:+.2f}) "
          f"margin={2 / np.linalg.norm(w):.2f} SV={len(s.support_):3d} "
          f"jumlah xi={xi.sum():5.1f} akurasi={s.score(X, ys):.3f}")
    if C in (0.01, 100):
        nama = "kecil" if C < 1 else "besar"
        with open(f"gambar/bab14/garis_{nama}.dat", "w") as f:
            f.write(f"w1 w2 b\n{w[0]:.4f} {w[1]:.4f} {b:.4f}\n")
        with open(f"gambar/bab14/titik_{nama}.dat", "w") as f:
            f.write("x1 x2 kelas sv\n")
            sv = set(s.support_)
            for i, (a, c) in enumerate(X):
                f.write(f"{a:.3f} {c:.3f} {int(ys[i] > 0)} "
                        f"{int(i in sv)}\n")
