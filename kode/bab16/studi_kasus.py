import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from pola_gerai import indeks_gerai

F, gerai = indeks_gerai()                                        #<1>
print(F.groupby(gerai["tipe_lokasi"]).mean().round(3).to_string())
lama = gerai.index[gerai["tanggal_buka"] < "2025-01-01"]         #<2>
POLA = ["libur_nasional", "libur_semester", "ramadan"]           #<3>
X = F.loc[lama, POLA]
sk = StandardScaler().fit(X)
Z = sk.transform(X)
d = pdist(Z)
print(f"\n{len(lama)} gerai dibuka sebelum 2025; korelasi kofenetik:")
for m in ["single", "complete", "average", "ward"]:              #<4>
    print(f"  {m:8s} {cophenet(linkage(Z, m), d)[0]:.3f}")
L = linkage(Z, "ward")
print("tinggi 6 penggabungan terakhir:", np.round(L[-6:, 2], 2))  #<5>
for k in range(2, 7):
    lab = fcluster(L, k, "maxclust")
    print(f"k={k}: silhouette {silhouette_score(Z, lab):.3f}, ARI vs "
          f"tipe lokasi {adjusted_rand_score(gerai.loc[lama, 'tipe_lokasi'], lab):.3f}")
kl = pd.Series(fcluster(L, 4, "maxclust"), index=lama, name="klaster")
print("\n" + pd.crosstab(kl, gerai.loc[lama, "tipe_lokasi"]).to_string())
print("\nprofil klaster (indeks, 1 = sama dengan hari biasa):")
print(F.loc[lama, POLA].groupby(kl).mean().round(3).to_string())
for g in lama:                                                   #<6>
    mayor = gerai.loc[kl.index[kl == kl[g]], "tipe_lokasi"].mode()[0]
    if gerai.loc[g, "tipe_lokasi"] != mayor:
        print(f"menyimpang: {g} {gerai.loc[g, 'nama_gerai']} "
              f"({gerai.loc[g, 'tipe_lokasi']}) di klaster {kl[g]}")
pusat = pd.DataFrame(Z, index=lama).groupby(kl).mean()           #<7>
for g in gerai.index.difference(lama):
    z = sk.transform(F.loc[[g], POLA])
    k = ((pusat.to_numpy() - z) ** 2).sum(1).argmin() + 1
    print(f"gerai baru {g} ({gerai.loc[g, 'tipe_lokasi']}, buka "
          f"{gerai.loc[g, 'tanggal_buka']}): terdekat ke klaster {k}")
np.save("kode/bab16/linkage_gerai.npy", L)
kl.to_csv("kode/bab16/klaster_gerai.csv")
