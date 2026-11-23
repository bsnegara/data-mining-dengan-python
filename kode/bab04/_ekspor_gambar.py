"""Data gambar Bab 4 untuk pgfplots."""
import numpy as np
import pandas as pd

DATA, G = "kode/data/", "gambar/bab04/"
pel = pd.read_csv(DATA + "pelanggan.csv")
x = pel["nilai_90h"] / 1e3
h, tepi = np.histogram(x, bins=40, range=(0, 2000))
pd.DataFrame({"x": tepi[:-1], "n": h}).to_csv(G + "hist_nilai.dat", sep=" ", index=False)
h, tepi = np.histogram(np.log10(pel["nilai_90h"] + 1), bins=40, range=(0, 7))
pd.DataFrame({"x": tepi[:-1], "n": h}).to_csv(G + "hist_lognilai.dat", sep=" ", index=False)
# kotak garis recency per churn
with open(G + "kotak_recency.tex", "w") as f:
    for c in [0, 1]:
        r = pel.loc[pel.churn == c, "recency_hari"]
        q1, me, q3 = r.quantile([.25, .5, .75])
        iqr = q3 - q1
        lo = r[r >= q1 - 1.5 * iqr].min(); hi = r[r <= q3 + 1.5 * iqr].max()
        f.write(f"\\addplot+[boxplot prepared={{median={me},upper quartile={q3},"
                f"lower quartile={q1},upper whisker={hi},lower whisker={lo}}}] coordinates {{}};\n")
# matriks korelasi
kol = ["frekuensi_90h", "recency_hari", "nilai_90h", "rata_nilai_struk", "porsi_promo",
       "porsi_aplikasi", "keluhan_90h", "umur"]
c = pel[kol].corr().values
with open(G + "korelasi.dat", "w") as f:
    f.write("x y r\n")
    for i in range(len(kol)):
        for j in range(len(kol)):
            f.write(f"{j} {i} {c[i, j]:.2f}\n")
        f.write("\n")
# diagram sebar frekuensi vs nilai (sampel)
s = pel.sample(600, random_state=1)
s[["frekuensi_90h", "nilai_90h", "churn"]].assign(nilai_90h=lambda d: d.nilai_90h / 1e3).to_csv(
    G + "sebar.dat", sep=" ", index=False)
print("ok")
h, tepi = np.histogram(pel["umur"], bins=24, range=(16, 64))
pd.DataFrame({"x": tepi[:-1], "n": h}).to_csv(G + "hist_umur.dat", sep=" ", index=False)
