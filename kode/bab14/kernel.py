import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

x, z = np.array([1.0, 2.0]), np.array([3.0, 0.5])
phi = lambda v: np.array([v[0]**2, v[1]**2, np.sqrt(2) * v[0] * v[1]])
print(f"(x.z)^2          = {(x @ z)**2:.2f}")                    #<1>
print(f"phi(x).phi(z)    = {phi(x) @ phi(z):.2f}")

X, y = make_circles(400, noise=0.08, factor=0.5, random_state=0)  #<2>
for nama, m in {"linier": SVC(kernel="linear"),
                "polinomial d=2": SVC(kernel="poly", degree=2, coef0=1),
                "RBF": SVC(kernel="rbf", gamma=1.0)}.items():
    s = cross_val_score(m, X, y, cv=5).mean()
    print(f"{nama:15s} akurasi CV {s:.3f}")

for g in [0.1, 1, 10, 100]:                                      #<3>
    m = SVC(kernel="rbf", gamma=g, C=1).fit(X, y)
    s = cross_val_score(SVC(kernel="rbf", gamma=g), X, y, cv=5).mean()
    print(f"RBF gamma={g:5g}: akurasi latih {m.score(X, y):.3f}, "
          f"CV {s:.3f}, SV {len(m.support_)}")
