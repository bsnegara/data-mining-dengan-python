import numpy as np
from hmmlearn.hmm import CategoricalHMM

pi0 = np.array([0.6, 0.4])                                       #<1>
A0 = np.array([[0.7, 0.3], [0.2, 0.8]])
B0 = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
data = [[0, 1, 2], [0, 0, 1], [1, 2, 2], [0, 1, 0]]  # mdk mmd dkk mdm


def e_step(o, pi, A, B):                                         #<2>
    T = len(o)
    a = np.zeros((T, 2)); b = np.ones((T, 2))
    a[0] = pi * B[:, o[0]]
    for t in range(1, T):
        a[t] = (a[t - 1] @ A) * B[:, o[t]]
    for t in range(T - 2, -1, -1):
        b[t] = A @ (B[:, o[t + 1]] * b[t + 1])
    p = a[-1].sum()
    g = a * b / p
    xi = [a[t][:, None] * A * (B[:, o[t + 1]] * b[t + 1]) / p
          for t in range(T - 1)]
    return g, np.sum(xi, 0), np.log(p)


def m_step(data, pi, A, B):                                      #<3>
    G0, Xi, Gt, Bn, ll = 0, 0, 0, np.zeros_like(B), 0
    for o in data:
        g, xi, lp = e_step(o, pi, A, B)
        G0 = G0 + g[0]; Xi = Xi + xi; Gt = Gt + g[:-1].sum(0)
        for t, s in enumerate(o):
            Bn[:, s] += g[t]
        ll += lp
    return (G0 / len(data), Xi / Gt[:, None],
            Bn / Bn.sum(1, keepdims=True), ll)


pi, A, B = pi0, A0, B0                                           #<4>
for it in range(1, 31):
    pi, A, B, ll = m_step(data, pi, A, B)
    if it in (1, 2, 3, 10, 30):
        print(f"iterasi {it:2d}: log-likelihood sebelum = {ll:.4f}")
    if it == 1:
        print("  pi =", pi.round(3), " A =", A.round(3).tolist())
        print("  B  =", B.round(3).tolist())
print("akhir: A =", A.round(3).tolist())
print("       B =", B.round(3).tolist())

m = CategoricalHMM(2, n_iter=30, tol=0, init_params="",          #<5>
                   params="ste")
m.startprob_, m.transmat_, m.emissionprob_ = pi0, A0, B0
X = np.concatenate(data).reshape(-1, 1)
m.fit(X, lengths=[len(o) for o in data])
print("hmmlearn A =", m.transmat_.round(3).tolist())
print("hmmlearn B =", m.emissionprob_.round(3).tolist())
