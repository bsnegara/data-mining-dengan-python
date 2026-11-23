import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from siapkan import muat

rng = np.random.default_rng(0)
X, y = make_moons(400, noise=0.07, random_state=0)               #<1>
X = np.vstack([X, rng.uniform([-1.5, -1], [2.5, 1.5], (20, 2))])
y = np.r_[y, -np.ones(20, int)]
km = KMeans(2, n_init=10, random_state=0).fit(X)
db = DBSCAN(eps=0.15, min_samples=8).fit(X)                      #<2>
print(f"bulan sabit: ARI k-means {adjusted_rand_score(y, km.labels_):.3f}"
      f", ARI DBSCAN {adjusted_rand_score(y, db.labels_):.3f}")
print(f"  DBSCAN: {db.labels_.max() + 1} klaster, "
      f"{(db.labels_ == -1).sum()} titik derau")
np.savetxt("gambar/bab15/bulan.dat", np.c_[X, km.labels_, db.labels_],
           fmt="%.3f %.3f %d %d", header="x y km db", comments="")
d, _ = NearestNeighbors(n_neighbors=8).fit(X).kneighbors(X)       #<3>
np.savetxt("gambar/bab15/k_jarak.dat",
           np.c_[np.arange(len(d)), np.sort(d[:, -1])],
           fmt="%d %.4f", header="i d", comments="")

df, Xa = muat()                                                  #<4>
Z = StandardScaler().fit_transform(Xa)
print("\nanggota (7 atribut terstandar), min_samples = 14:")
for eps in [0.6, 0.7, 0.8, 1.0]:
    lab = DBSCAN(eps=eps, min_samples=14).fit(Z).labels_
    besar = sorted(np.bincount(lab[lab >= 0]), reverse=True)[:3]
    print(f"  eps={eps:.1f}: {lab.max() + 1:2d} klaster, derau "
          f"{(lab == -1).mean():5.1%}, tiga terbesar {[int(b) for b in besar]}")
