import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
data = {"AND": np.array([0, 0, 0, 1]), "XOR": np.array([0, 1, 1, 0])}


def latih(X, y, eta=0.5, epoch=20, cetak=False):
    w, b = np.zeros(2), 0.0                                      #<1>
    for e in range(1, epoch + 1):
        salah = 0
        for xi, yi in zip(X, y):
            yhat = int(w @ xi + b > 0)                           #<2>
            if yhat != yi:
                w += eta * (yi - yhat) * xi                      #<3>
                b += eta * (yi - yhat)
                salah += 1
            if cetak and e <= 2:
                print(f"  x={xi} y={yi} yhat={yhat} -> w={w} b={b:+.1f}")
        if salah == 0:
            return w, b, e
    return w, b, None


for nama, y in data.items():
    print(nama)
    w, b, e = latih(X, y, cetak=(nama == "AND"))
    if e:
        print(f"  konvergen pada epoch {e}: w={w}, b={b:+.1f}")
    else:
        print("  tidak konvergen setelah 20 epoch")              #<4>
