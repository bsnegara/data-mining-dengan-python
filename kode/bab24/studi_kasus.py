import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from siapkan import KEJADIAN, SELESAI, muat, pad
from hmm_beralih import latih, maju, mundur

seq, info, niat = muat()
O, M = pad(list(seq[info.latih]))
Ou, Mu = pad(list(seq[~info.latih]))

hasil = [latih(O, M, S=2, seed=s) for s in range(3)]            #<1>
for s, h in enumerate(hasil):
    print(f"awal acak {s}: log-lik latih {h[3]:.0f}")
pi, A, E, _ = max(hasil, key=lambda h: h[3])
ll_uji = np.log(maju(Ou, Mu, pi, A, E)[1]).sum()
print(f"log-lik uji HMM beralih: {ll_uji:.0f}")

nt = int(np.argmax(E[:, 5, 6]))     # keranjang -> bayar           #<2>
jl = 1 - nt
print(f"\nkeadaan niat = {nt}; pi(niat) = {pi[nt]:.2f}")
print(f"peluang tetap: jelajah {A[jl, jl]:.3f}, "
      f"niat {A[nt, nt]:.3f}")
K = {k: i for i, k in enumerate(KEJADIAN)}
for a, b in [("beranda", "promo"), ("beranda", "cari"),
             ("lihat_menu", "detail_menu"), ("detail_menu",
             "keranjang"), ("keranjang", "bayar"),
             ("keranjang", "keluar"), ("bayar", "selesai")]:
    print(f"  {a:>11s} -> {b:<11s} jelajah {E[jl, K[a], K[b]]:.2f}"
          f"   niat {E[nt, K[a], K[b]]:.2f}")

a, c = maju(Ou, Mu, pi, A, E)                                    #<3>
g = a * mundur(Ou, Mu, A, E, c)
benar = np.zeros_like(Ou)
for i, z in enumerate(niat[~info.latih]):
    benar[i, :len(z)] = z
for nama, p in [("penyaringan", a), ("penghalusan", g)]:
    tepat = ((p[..., nt] > .5) == benar)[Mu].mean()
    print(f"kecocokan {nama} dengan niat_laten: {tepat:.3f}")


def serap(P):                                                    #<4>
    P = P.copy()
    P[SELESAI] = P[SELESAI + 1] = 0
    P[SELESAI, SELESAI] = P[SELESAI + 1, SELESAI + 1] = 1
    N = np.linalg.inv(np.eye(7) - P[:7, :7])
    return (N @ P[:7, 7:])[:, 0]


n = np.zeros((9, 9))
for s in seq[info.latih]:
    np.add.at(n, (s[:-1], s[1:]), 1)
n[7:, :] += np.eye(9)[7:]
b_markov = serap(n / n.sum(1, keepdims=True))
b_rezim = np.array([serap(E[j]) for j in range(2)])


def ciri(s, k):
    hit = np.bincount(s[:k], minlength=9)[:7]
    return np.r_[hit, np.eye(7)[s[k - 1]]]


print("\nAUC menebak pembelian setelah k kejadian (data uji):")  #<5>
print("  k  sesi  beli  halaman  logistik  HMM-niat  HMM-beli")
uji, lat = seq[~info.latih], seq[info.latih]
auc = []
for k in (2, 3, 4, 5, 6):
    idx = [i for i, s in enumerate(uji) if len(s) > k
           and s[k - 1] < SELESAI]
    y = np.array([uji.iloc[i][-1] == SELESAI for i in idx])
    hal = b_markov[[uji.iloc[i][k - 1] for i in idx]]
    tr = [s for s in lat if len(s) > k and s[k - 1] < SELESAI]
    lr = LogisticRegression(max_iter=2000).fit(
        [ciri(s, k) for s in tr], [s[-1] == SELESAI for s in tr])
    plr = lr.predict_proba([ciri(uji.iloc[i], k) for i in idx])[:, 1]
    pn = a[idx, k - 1, nt]
    pb = (a[idx, k - 1] * b_rezim[:, Ou[idx, k - 1]].T).sum(1)
    baris = [roc_auc_score(y, p) for p in (hal, plr, pn, pb)]
    auc.append([k] + baris)
    print(f"  {k}  {len(idx):4d}  {y.mean():.2f}  " +
          "  ".join(f"{v:7.3f}" for v in baris))
pd.DataFrame(auc, columns=["k", "hal", "lr", "niat", "beli"]).to_csv(
    "gambar/bab24/auc.dat", sep=" ", index=False, float_format="%.3f")
np.savez("kode/bab24/_model.npz", pi=pi, A=A, E=E, nt=nt,
         b_markov=b_markov)
