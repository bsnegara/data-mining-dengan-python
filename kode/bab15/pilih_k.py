from sklearn.cluster import KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler
from siapkan import muat

df, X = muat()
Z = StandardScaler().fit_transform(X)
print(" k   inersia  silhouette  Calinski-H  Davies-B")
with open("gambar/bab15/pilih_k.dat", "w") as f:
    f.write("k inersia siluet\n")
    for k in range(2, 10):
        km = KMeans(k, n_init=10, random_state=0).fit(Z)
        s = silhouette_score(Z, km.labels_, sample_size=4000,
                             random_state=0)                     #<1>
        ch = calinski_harabasz_score(Z, km.labels_)
        db = davies_bouldin_score(Z, km.labels_)
        print(f"{k:2d} {km.inertia_:9.0f} {s:11.3f} {ch:11.0f} {db:9.3f}")
        f.write(f"{k} {km.inertia_:.0f} {s:.4f}\n")
