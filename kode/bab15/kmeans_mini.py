import numpy as np

nama = np.array(list("ABCDEFGHIJ"))
X = np.array([[2, 1.0], [3, 1.5], [4, 1.0], [10, 3.0], [12, 3.5],    #<1>
              [11, 2.5], [6, 6.0], [7, 7.0], [5, 6.5], [8, 6.0]])


def kmeans(pusat, cetak=True):
    for it in range(1, 20):
        d = np.linalg.norm(X[:, None, :] - pusat[None, :, :], axis=2)  #<3>
        label = d.argmin(1)
        sse = (d.min(1) ** 2).sum()
        baru = np.array([X[label == k].mean(0) for k in range(3)])  #<4>
        if cetak:
            print(f"iterasi {it}: klaster", "".join(str(l + 1)
                                                    for l in label),
                  f" SSE {sse:.2f}")
            print("  pusat baru:", "  ".join(f"({a:.2f}; {b:.2f})"
                                             for a, b in baru))
        if np.allclose(baru, pusat):
            return label, sse
        pusat = baru


label, sse = kmeans(X[[0, 1, 2]].astype(float))                  #<2>
print("konvergen; SSE akhir", round(sse, 2))
for awal in ["DEF", "AGH"]:                                      #<5>
    idx = [list(nama).index(c) for c in awal]
    label, sse = kmeans(X[idx].astype(float), cetak=False)
    print(f"pusat awal {awal}: klaster",
          " ".join("".join(nama[label == k]) for k in range(3)),
          f" SSE {sse:.2f}")
