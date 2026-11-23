import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

d = pd.read_csv("kode/data/profil_rasa.csv")                     #<1>
ATR = d.columns[3:]
X = StandardScaler().fit_transform(d[ATR])
print(f"{len(d)} kopi x {len(ATR)} atribut rasa")
R = np.corrcoef(X.T)
print(f"korelasi |r|>0,6 antaratribut: "
      f"{(np.abs(R[np.triu_indices(12, 1)]) > .6).sum()} dari 66")

pca = PCA().fit(X)                                               #<2>
lam = pca.explained_variance_
rng = np.random.default_rng(20)                                  #<3>
acak = np.array([PCA().fit(rng.standard_normal(X.shape))
                 .explained_variance_ for _ in range(500)])
batas = np.percentile(acak, 95, axis=0)
print("\nkomp  nilai eigen  porsi  kumulatif  batas acak 95%")
for i in range(6):
    print(f"PC{i + 1:<3d} {lam[i]:10.3f}  {lam[i] / lam.sum():.3f}"
          f"  {lam[:i + 1].sum() / lam.sum():9.3f}  {batas[i]:13.3f}")
k = int((lam > batas).sum())
print(f"komponen di atas batas acak: {k}")

pca3 = PCA(n_components=3).fit(X)                                #<4>
Z = pca3.transform(X)
# arah tanda: kekentalan, rasa ikutan, dan manis dibuat positif
acuan = [9, 11, 8]
arah = np.sign([np.corrcoef(X[:, j], Z[:, c])[0, 1]
                for c, j in enumerate(acuan)])
Z = Z * arah
L = pd.DataFrame([[np.corrcoef(X[:, j], Z[:, c])[0, 1]
                   for c in range(3)] for j in range(12)],
                 index=ATR, columns=["PC1", "PC2", "PC3"])
print("\nloading (korelasi atribut-komponen):")
print(L.round(2).to_string())
r2 = (L ** 2).sum(1)                                             #<5>
print("\nR^2 atribut dari 3 komponen, min/median:",
      f"{r2.min():.2f} / {r2.median():.2f}")
print("R^2 atribut dari 2 komponen, min/median:",
      f"{(L.iloc[:, :2] ** 2).sum(1).min():.2f} / "
      f"{(L.iloc[:, :2] ** 2).sum(1).median():.2f}")

asli = pca3.components_.T                                        #<6>
kongruen = []
for _ in range(500):
    i = rng.integers(0, len(X), len(X))
    b = PCA(n_components=3).fit(X[i]).components_.T
    kongruen.append(np.abs((asli * b).sum(0)))
kongruen = np.array(kongruen)
print("kemiripan loading bootstrap (median, persentil 5):")
for j in range(3):
    print(f"  PC{j + 1}: {np.median(kongruen[:, j]):.3f}"
          f"  {np.percentile(kongruen[:, j], 5):.3f}")

skor = pd.DataFrame(Z[:, :2], columns=["berat", "ikutan"])       #<7>
skala = (1 + 4 * (skor - skor.min()) / (skor.max() - skor.min()))
hasil = pd.concat([d[["kopi", "spesies"]], skor.round(2),
                   skala.round().astype(int).add_suffix("_15")],
                  axis=1)
print("\ncontoh kartu menu (skala 1-5):")
pilih = ["Kintamani", "Kerinci", "Bajawa", "Toraja", "Gayo",
         "Mandailing", "Lampung Robusta"]
print(hasil.set_index("kopi").loc[pilih].iloc[:, 1:].to_string())

with open("gambar/bab20/skor.dat", "w") as f:
    f.write("pc1 pc2 rob nama\n")
    for _, r in hasil.iterrows():
        f.write(f"{r.berat:.3f} {r.ikutan:.3f} "
                f"{int(r.spesies == 'robusta')} {{{r.kopi}}}\n")
with open("gambar/bab20/loading.dat", "w") as f:
    f.write("l1 l2 nama\n")
    for a, r in L.iterrows():
        f.write(f"{r.PC1:.3f} {r.PC2:.3f} {{{a}}}\n")
with open("gambar/bab20/scree.dat", "w") as f:
    f.write("k lam kum acak\n")
    for i in range(12):
        f.write(f"{i + 1} {lam[i]:.3f} {lam[:i + 1].sum() / 12:.3f}"
                f" {batas[i]:.3f}\n")
