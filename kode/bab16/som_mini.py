import numpy as np

W = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])               #<1>
X = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.7], [0.8, 0.3]])
eta, r = 0.5, 1                                                  #<2>
for t, x in enumerate(X[:2], 1):
    d = np.linalg.norm(W - x, axis=1)
    j = d.argmin()                                               #<3>
    print(f"masukan {t} x={x}: jarak {d.round(3)}, pemenang neuron {j + 1}")
    for i in range(3):
        if abs(i - j) <= r:                                      #<4>
            h = 1.0 if i == j else 0.5
            W[i] += eta * h * (x - W[i])
    print("  bobot baru:", "  ".join(f"({a:.3f}; {b:.3f})" for a, b in W))


def latih(X, W, n=200, eta0=0.5, r0=1.0, seed=0):                #<5>
    rng = np.random.default_rng(seed)
    for t in range(n):
        x = X[rng.integers(len(X))]
        eta = eta0 * (1 - t / n)
        r = r0 * (1 - t / n)
        j = np.linalg.norm(W - x, axis=1).argmin()
        for i in range(len(W)):
            h = np.exp(-((i - j) ** 2) / (2 * max(r, 0.1) ** 2))
            W[i] += eta * h * (x - W[i])
    return W


print("setelah 200 langkah:",
      "  ".join(f"({a:.2f}; {b:.2f})" for a, b in latih(X, W)))
