import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

d = pd.read_csv("kode/data/mutu_biji.csv")
KOL = ["ketinggian_mdpl", "kadar_air", "densitas_gl",
       "ukuran_screen", "cacat_primer", "cacat_sekunder"]
X = d[KOL]
hasil = {"simp. baku": X.std()}
for nama, data in [("PC1 mentah", X),                            #<1>
                   ("PC1 baku", StandardScaler().fit_transform(X))]:
    pca = PCA().fit(data)
    hasil[nama] = pd.Series(pca.components_[0], KOL)             #<2>
    rasio = pca.explained_variance_ratio_[:3].round(3)
    print(f"{nama:10s}: porsi varians 3 komponen pertama {rasio}")
print((pd.DataFrame(hasil).round(3) + 0.0).to_string())
