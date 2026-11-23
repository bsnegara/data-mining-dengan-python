import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from siapkan import PERILAKU, muat

df, X = muat()
sk = StandardScaler().fit(X)
Z = sk.transform(X)
km = KMeans(5, n_init=20, random_state=0).fit(Z)                 #<1>
df["klaster"] = km.labels_

prof = df.groupby("klaster").agg(                                #<2>
    n=("id_anggota", "size"), frek=("frekuensi_90h", "mean"),
    struk=("rata_nilai_struk", "mean"), pagi=("porsi_pagi", "mean"),
    akhir_pkn=("porsi_akhir_pekan", "mean"), promo=("porsi_promo", "mean"),
    hitam=("porsi_kopi_hitam", "mean"), apl=("porsi_aplikasi", "mean"),
    umur=("umur", "mean"), churn=("churn", "mean"))
prof["struk"] = prof["struk"] / 1000
print(prof.round(2).to_string())

s = silhouette_samples(Z, km.labels_)                            #<3>
print("\nsilhouette rata-rata per klaster:", "  ".join(
    f"{k}: {v:.3f}" for k, v in pd.Series(s).groupby(km.labels_).mean().items()))

stab = []                                                        #<4>
rng = np.random.default_rng(0)
for b in range(20):
    i = rng.choice(len(Z), len(Z), replace=True)
    kb = KMeans(5, n_init=5, random_state=b).fit(Z[i])
    stab.append(adjusted_rand_score(km.labels_, kb.predict(Z)))
print(f"stabilitas bootstrap (ARI): rata-rata {np.mean(stab):.3f}, "
      f"minimum {np.min(stab):.3f}")

z = pd.DataFrame(km.cluster_centers_, columns=PERILAKU)          #<5>
with open("gambar/bab15/pusat_grid.dat", "w") as f:
    for i in range(len(z)):
        for j in range(z.shape[1]):
            f.write(f"{j} {i} {z.iloc[i, j]:.2f}\n")
        f.write("\n")
print("\nsilang dengan tingkat keanggotaan (persen baris):")
print((pd.crosstab(df["klaster"], df["tingkat"], normalize="index")
       * 100).round(0).to_string())
print(f"\nARI terhadap segmen laten (hanya ada di data sintetis): "
      f"{adjusted_rand_score(df['segmen_laten'], km.labels_):.3f}")  #<6>
