import sys; sys.path.insert(0, "kode/bab19")
import pandas as pd
from jb import JaringanBayes, muat_biji

d = muat_biji()
hit = pd.crosstab([d.air, d.simpan], d.jamur)                    #<1>
print("hitungan kiriman menurut induk jamur:")
print(hit.to_string())
mle = hit["ya"] / hit.sum(1)                                     #<2>
lap = (hit["ya"] + 1) / (hit.sum(1) + 2)
print("\nP(jamur=ya | air, simpan):")
tabel = pd.DataFrame({"MLE": mle, "Laplace": lap}).round(3)
print(tabel.to_string())

print("\nCPT cacat dengan dua induk (jamur, jemur):")            #<3>
V = ["jamur", "jemur", "cacat"]
jb = JaringanBayes({"jamur": [], "jemur": [],
                    "cacat": ["jamur", "jemur"]},
                   {v: sorted(d[v].unique()) for v in V}).latih(d)
for (jm, je), t in jb.cpt["cacat"].items():
    n = ((d.jamur == jm) & (d.jemur == je)).sum()
    print(f"  jamur={jm:5s} jemur={je:9s} n={n:4d}"
          f"  P(cacat=ya)={t['ya']:.3f}")
