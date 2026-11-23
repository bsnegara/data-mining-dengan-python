import sys; sys.path.insert(0, "kode/bab15")
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler
from siapkan import muat

df, X = muat()                                                   #<1>
Z = StandardScaler().fit_transform(X)
seg = KMeans(5, n_init=20, random_state=0).fit_predict(Z)
rng = np.random.default_rng(21)
i = rng.choice(len(Z), 2000, replace=False)                      #<2>
Z, seg = Z[i], seg[i]

peta = {"PCA": PCA(2).fit_transform(Z)}                          #<3>
for p in [5, 30, 100]:
    peta[f"t-SNE p={p}"] = TSNE(perplexity=p, init="pca",
                                random_state=0).fit_transform(Z)
print("metode        trustworthiness (k=10)")                    #<4>
for nama, Y in peta.items():
    print(f"{nama:13s} {trustworthiness(Z, Y, n_neighbors=10):.3f}")

with open("gambar/bab21/tsne.dat", "w") as f:
    f.write("px py tx ty seg\n")
    for a, b, s in zip(peta["PCA"], peta["t-SNE p=30"], seg):
        f.write(f"{a[0]:.3f} {a[1]:.3f} {b[0]:.2f} {b[1]:.2f} {s}\n")
