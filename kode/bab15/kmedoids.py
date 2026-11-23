import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def kmedoids(D, k, iterasi=50, seed=0):                          #<1>
    rng = np.random.default_rng(seed)
    med = rng.choice(len(D), k, replace=False)
    for _ in range(iterasi):
        label = D[:, med].argmin(1)
        baru = med.copy()
        for j in range(k):
            anggota = np.where(label == j)[0]
            biaya = D[np.ix_(anggota, anggota)].sum(1)           #<2>
            baru[j] = anggota[biaya.argmin()]
        if set(baru) == set(med):
            break
        med = baru
    return med, D[:, med].argmin(1)


rng = np.random.default_rng(1)
X = np.vstack([rng.normal([0, 0], 0.5, (30, 2)),
               rng.normal([4, 0], 0.5, (30, 2))])
X_pencilan = np.vstack([X, [[40, 0]]])                           #<3>
for nama, data in [("tanpa pencilan", X), ("dengan pencilan", X_pencilan)]:
    km = KMeans(2, n_init=10, random_state=0).fit(data)
    D = pairwise_distances(data, metric="manhattan")
    med, _ = kmedoids(D, 2)
    print(f"{nama}:")
    print("  pusat k-means :", np.round(sorted(km.cluster_centers_[:, 0]), 2))
    print("  medoid        :", np.round(sorted(data[med, 0]), 2))
