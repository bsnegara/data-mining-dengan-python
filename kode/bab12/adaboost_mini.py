import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([1, 1, -1, -1, 1, 1, 1, -1])                       #<1>
w = np.full(8, 1 / 8)


def tunggul_terbaik(x, y, w):                                    #<2>
    terbaik = None
    for t in np.arange(1.5, 8, 1.0):
        for arah in [1, -1]:
            h = np.where(x < t, arah, -arah)
            e = w[h != y].sum()
            if terbaik is None or e < terbaik[0] - 1e-12:
                terbaik = (e, t, arah, h)
    return terbaik


F = np.zeros(8)
for m in range(1, 4):
    e, t, arah, h = tunggul_terbaik(x, y, w)
    alfa = 0.5 * np.log((1 - e) / e)                             #<3>
    print(f"putaran {m}: h(x) = {arah:+d} jika x < {t}, "
          f"selain itu {-arah:+d}")
    print(f"  galat berbobot e = {e:.4f}, alfa = {alfa:.4f}")
    w = w * np.exp(-alfa * y * h)                                #<4>
    w = w / w.sum()
    print("  bobot baru:", np.round(w, 4))
    F += alfa * h
    print("  skor F:", np.round(F, 3), "-> salah",
          int((np.sign(F) != y).sum()))
