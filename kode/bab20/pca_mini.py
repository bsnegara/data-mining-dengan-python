import numpy as np
from sklearn.decomposition import PCA

X = np.array([[4, 5], [5, 4], [6, 6], [7, 8], [8, 7]], float)    #<1>
print("rata-rata:", X.mean(0))
Xc = X - X.mean(0)                                               #<2>
S = Xc.T @ Xc / (len(X) - 1)
print("matriks kovarians:\n", S)
nilai, vektor = np.linalg.eigh(S)                                #<3>
urut = np.argsort(nilai)[::-1]
nilai, vektor = nilai[urut], vektor[:, urut]
print("nilai eigen:", nilai.round(4))
print("vektor eigen (kolom):\n", vektor.round(4))
print("porsi varians:", (nilai / nilai.sum()).round(3))
Z = Xc @ vektor                                                  #<4>
print("skor komponen:\n", Z.round(3))
pca = PCA().fit(X)                                               #<5>
print("sklearn komponen (baris):\n", pca.components_.round(4))
print("sklearn varians:", pca.explained_variance_.round(4))
