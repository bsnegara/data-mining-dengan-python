import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM
from siapkan import KEJADIAN, SELESAI, muat
warnings.simplefilter("ignore")

seq, info, _ = muat()                                            #<1>
print(f"{len(seq)} sesi, {info.panjang.sum()} kejadian; "
      f"latih {info.latih.sum()}, uji {(~info.latih).sum()}")
print(f"proporsi beli {info.beli.mean():.3f}; median panjang "
      f"{info.panjang.median():.0f} kejadian")
latih, uji = seq[info.latih], seq[~info.latih]

V = len(KEJADIAN)                                                #<2>
n = np.zeros((V, V))
for s in latih:
    np.add.at(n, (s[:-1], s[1:]), 1)
n[SELESAI, SELESAI] = n[SELESAI + 1, SELESAI + 1] = 1
P = n / n.sum(1, keepdims=True)
Q, R = P[:7, :7], P[:7, 7:]
N = np.linalg.inv(np.eye(7) - Q)
tab = pd.DataFrame({"p_keluar_langsung": P[:7, 8],
                    "p_beli_akhirnya": (N @ R)[:, 0],
                    "sisa_langkah": N.sum(1)}, index=KEJADIAN[:7])
print("\nRantai Markov (data latih):")
print(tab.round(2).to_string())


def ll_markov(daftar):                                           #<3>
    return sum(np.log(P[s[:-1], s[1:]]).sum() for s in daftar)


print(f"\nlog-likelihood uji rantai Markov: {ll_markov(uji):.0f}")

X = np.concatenate(latih.values).reshape(-1, 1)                  #<4>
L = latih.map(len).to_numpy()
Xu = np.concatenate(uji.values).reshape(-1, 1)
Lu = uji.map(len).to_numpy()
hmm = {}
for S in (2, 3, 4):
    hmm[S] = CategoricalHMM(S, n_iter=100, tol=0.5, random_state=1,
                            n_features=V).fit(X, L)
    print(f"HMM biasa {S} keadaan: log-lik uji "
          f"{hmm[S].score(Xu, Lu):.0f}, BIC {hmm[S].bic(X, L):.0f}")

print("\nemisi tertinggi HMM biasa 3 keadaan:")                #<5>
for j, b in enumerate(hmm[3].emissionprob_):
    top = np.argsort(b)[::-1][:3]
    print(f"  keadaan {j}: " + ", ".join(
        f"{KEJADIAN[k]} {b[k]:.2f}" for k in top))
print("  transisi:", hmm[3].transmat_.round(2).tolist())
