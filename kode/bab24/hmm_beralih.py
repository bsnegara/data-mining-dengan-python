"""HMM dengan emisi Markov: P(o_t | o_t-1, z_t) = E[z_t, o_t-1, o_t].
Semua sesi diproses serentak (tervektorisasi) dengan penskalaan."""
import numpy as np


def maju(O, M, pi, A, E):                                        #<1>
    n, T = O.shape
    a = np.zeros((n, T, len(pi)))
    c = np.ones((n, T))
    a[:, 0] = pi                     # kejadian pertama selalu beranda
    for t in range(1, T):
        x = (a[:, t - 1] @ A) * E[:, O[:, t - 1], O[:, t]].T
        m = M[:, t]
        c[m, t] = x[m].sum(1)
        a[m, t] = x[m] / c[m, t, None]
        a[~m, t] = a[~m, t - 1]      # sesi sudah berakhir
    return a, c


def mundur(O, M, A, E, c):                                       #<2>
    n, T = O.shape
    b = np.ones((n, T, A.shape[0]))
    for t in range(T - 2, -1, -1):
        e = E[:, O[:, t], O[:, t + 1]].T
        x = ((e * b[:, t + 1]) @ A.T) / c[:, t + 1, None]
        m = M[:, t + 1]
        b[m, t] = x[m]
    return b


def latih(O, M, S=2, iterasi=60, seed=0, V=9):                   #<3>
    r = np.random.default_rng(seed)
    pi = r.dirichlet(np.ones(S))
    A = np.full((S, S), 0.1 / (S - 1))
    np.fill_diagonal(A, 0.9)
    E = r.dirichlet(np.ones(V), (S, V))
    for _ in range(iterasi):
        a, c = maju(O, M, pi, A, E)                              # E
        b = mundur(O, M, A, E, c)
        g = a * b
        An = np.zeros((S, S))
        En = np.full((S, V, V), 1e-3)     # penghalusan kecil
        for t in range(1, O.shape[1]):                           # M
            m = M[:, t]
            e = E[:, O[:, t - 1], O[:, t]].T
            xi = (a[m, t - 1, :, None] * A
                  * (e[m] * b[m, t])[:, None, :])
            xi /= c[m, t, None, None]
            An += xi.sum(0)
            for s in range(S):
                np.add.at(En[s], (O[m, t - 1], O[m, t]), g[m, t, s])
        pi = g[:, 0].mean(0)
        A = An / An.sum(1, keepdims=True)
        E = En / En.sum(2, keepdims=True)
    loglik = np.log(maju(O, M, pi, A, E)[1]).sum()
    return pi, A, E, loglik
