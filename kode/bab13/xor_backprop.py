import numpy as np

sig = lambda z: 1 / (1 + np.exp(-z))
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
y = np.array([0, 1, 1, 0], float)
W1 = np.array([[0.5, -0.5], [0.5, -0.5]])                        #<1>
b1 = np.array([-0.2, 0.7])
w2 = np.array([1.0, 1.0])
b2 = -1.0
eta = 0.5

x, t = X[2], y[2]                                                #<2>
z1 = x @ W1 + b1; h = sig(z1)                                    #<3>
z2 = h @ w2 + b2; o = sig(z2)
print("maju : z1 =", z1.round(4), " h =", h.round(4))
print(f"       z2 = {z2:.4f}  o = {o:.4f}  galat = {0.5*(t-o)**2:.4f}")
d2 = (o - t) * o * (1 - o)                                       #<4>
d1 = d2 * w2 * h * (1 - h)                                       #<5>
print(f"mundur: delta2 = {d2:.4f}  delta1 =", d1.round(4))
w2 -= eta * d2 * h; b2 -= eta * d2                               #<6>
W1 -= eta * np.outer(x, d1); b1 -= eta * d1
print("baru : w2 =", w2.round(4), f" b2 = {b2:.4f}")
print("       W1 =", W1.round(4).tolist(), " b1 =", b1.round(4))
o_baru = sig(sig(x @ W1 + b1) @ w2 + b2)
print(f"       o baru = {o_baru:.4f} (target {t:.0f})")

rng = np.random.default_rng(3)                                   #<7>
W1, b1 = rng.normal(0, 1, (2, 2)), np.zeros(2)
w2, b2 = rng.normal(0, 1, 2), 0.0
for ep in range(1, 5001):
    H = sig(X @ W1 + b1); O = sig(H @ w2 + b2)
    D2 = (O - y) * O * (1 - O)
    D1 = np.outer(D2, w2) * H * (1 - H)
    w2 -= eta * H.T @ D2; b2 -= eta * D2.sum()
    W1 -= eta * X.T @ D1; b1 -= eta * D1.sum(0)
    if ep in (1, 1000, 5000):
        print(f"epoch {ep:4d}: galat {0.5*((y-O)**2).sum():.4f}"
              f"  o = {O.round(2)}")
