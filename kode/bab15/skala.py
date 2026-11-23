import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from siapkan import muat

df, X_mentah = muat(log=False)
_, X_log = muat(log=True)
for nama, X in [("mentah", X_mentah.to_numpy()),                 #<1>
                ("terstandar", StandardScaler().fit_transform(X_mentah)),
                ("log + terstandar", StandardScaler().fit_transform(X_log))]:
    km = KMeans(4, n_init=10, random_state=0).fit(X)
    ari = adjusted_rand_score(df["segmen_laten"], km.labels_)    #<2>
    ukuran = sorted(np.bincount(km.labels_).tolist() if False else
                    [int((km.labels_ == k).sum()) for k in range(4)])
    print(f"{nama:17s} ARI {ari:.3f}  ukuran klaster {ukuran}")
