import sys; sys.path.insert(0, "kode/bab23")
import numpy as np
from statsmodels.tsa.seasonal import STL
from siapkan import harian, mingguan

h = harian()                                                     #<1>
hari = (h.biji / h.gerai)[h.tutup == 0]
pola = hari.groupby(hari.index.dayofweek).mean()
print("biji per gerai per hari menurut hari (Senin=0):")
print("  " + "  ".join(f"{v:.2f}" for v in pola))

w = mingguan()                                                   #<2>
stl = STL(w.per_gerai, period=52, robust=True).fit()
kuat_tren = 1 - stl.resid.var() / (stl.trend + stl.resid).var()  #<3>
kuat_mus = 1 - stl.resid.var() / (stl.seasonal + stl.resid).var()
print(f"\nSTL mingguan: kekuatan tren {kuat_tren:.2f}, "
      f"kekuatan musiman {kuat_mus:.2f}")
tren = stl.trend.resample("YE").mean()
print("tren rata-rata per tahun:",
      ", ".join(f"{t.year} {v:.2f}" for t, v in tren.items()))
besar = stl.resid.abs().nlargest(4)                              #<4>
print("sisaan terbesar (minggu berakhir):")
for t, v in besar.items():
    print(f"  {t.date()}  {stl.resid[t]:+.2f}  ramadan {w.ramadan[t]}"
          f" tutup {w.tutup[t]} pasca {w.pasca[t]}")
with open("gambar/bab23/stl.dat", "w") as f:
    f.write("t y tren mus sisa\n")
    for i, t in enumerate(w.index):
        f.write(f"{t.year + (t.dayofyear - 1) / 365:.3f} "
                f"{w.per_gerai[t]:.3f} {stl.trend[t]:.3f} "
                f"{stl.seasonal[t]:.3f} {stl.resid[t]:.3f}\n")
