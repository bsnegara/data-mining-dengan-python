import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from siapkan import konten, muat

R, T, menu = muat()                                              #<1>
kode = list(menu.index)
u = R.index.intersection(T.index)
A = R.loc[u].to_numpy(float)
lama = A > 0
baru = (T.loc[u].to_numpy() > 0) & ~lama                         #<2>
ada = baru.sum(1) > 0
X = np.log1p(R.to_numpy(float))
pop = R.sum().to_numpy(float)
print(f"{len(R)} anggota latih (Jan-Sep 2025), {ada.sum()} anggota uji "
      f"memesan menu baru di Okt-Des")


def nilai(S, k=3, baris=None):
    S = S.astype(float)
    S[lama] = -np.inf                                            #<3>
    top = np.argsort(-S, 1)[:, :k]
    hit = np.take_along_axis(baru, top, 1)
    m = ada if baris is None else ada & baris
    return dict(presisi=hit[m].sum(1).mean() / k,
                recall=(hit[m].sum(1) / baru[m].sum(1)).mean(),
                hit=(hit[m].sum(1) > 0).mean(),
                pop_rata=pop[top[m]].mean() / pop.mean())


def ternorma(M, sumbu):
    return M / np.maximum(np.linalg.norm(M, axis=sumbu, keepdims=True),
                          1e-12)


skor = {}
skor["popularitas"] = np.tile(pop, (len(u), 1))                  #<4>
C = ternorma(konten(kode), 1)                                    #<5>
profil = ternorma(np.log1p(A) @ C, 1)
skor["konten"] = profil @ C.T
Si = ternorma(X, 0).T @ ternorma(X, 0)                           #<6>
np.fill_diagonal(Si, 0)
skor["item-kNN"] = np.log1p(A) @ Si
Su = ternorma(np.log1p(A), 1) @ ternorma(X, 1).T                 #<7>
tet = np.argsort(-Su, 1)[:, 1:51]
skor["user-kNN"] = np.array([(Su[i, tet[i]][:, None] *
                              (X[tet[i]] > 0)).sum(0)
                             for i in range(len(u))])
sv = TruncatedSVD(2, random_state=0).fit(X)                      #<8>
skor["SVD k=2"] = sv.transform(np.log1p(A)) @ sv.components_
skor["gabungan"] = sum(np.argsort(np.argsort(skor[n], 1), 1)     #<9>
                       for n in ["item-kNN", "SVD k=2", "konten"])

tab = pd.DataFrame({n: nilai(s) for n, s in skor.items()}).T
print("\nsemua anggota uji, top-3:")
print(tab.round(3).to_string())
sedikit = A.sum(1) <= 3                                          #<10>
print(f"\nanggota dengan <= 3 pesanan latih ({(sedikit & ada).sum()} orang):")
print(pd.DataFrame({n: nilai(s, baris=sedikit) for n, s in skor.items()})
      .T[["presisi", "hit"]].round(3).to_string())
