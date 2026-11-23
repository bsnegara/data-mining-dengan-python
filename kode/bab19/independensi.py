import sys; sys.path.insert(0, "kode/bab19")
import pandas as pd
from jb import muat_biji

d = muat_biji()
def porsi(df, kolom, target, nilai="ya"):
    kena = df[target] == nilai
    return kena.groupby([df[c] for c in kolom]).mean().round(3)

print("1) P(cacat=ya | musim): musim dan cacat bergantung")      #<1>
print(porsi(d, ["musim"], "cacat").to_string())
print("\n2) P(cacat=ya | jamur, jemur, musim): bila jamur dan")  #<2>
print("   jemur diketahui, musim hampir tak menambah informasi")
t = porsi(d, ["jamur", "jemur", "musim"], "cacat").unstack()
print(t.to_string())
print("\n3) kolider: lama simpan -> jamur <- kadar air")         #<3>
m = pd.read_csv("kode/data/mutu_biji.csv")
def r(df): return df.kadar_air.corr(df.lama_simpan_hari)
jm = m[m.jamur == 1]
print("   korelasi kadar air dan lama simpan")
print(f"   semua kiriman    (n={len(m)}): {r(m):+.3f}")
print(f"   kiriman berjamur (n={len(jm)}) : {r(jm):+.3f}")
