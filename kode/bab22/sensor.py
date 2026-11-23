import numpy as np
import pandas as pd

s = pd.read_csv("kode/data/sensor_sangrai.csv")
b = pd.read_csv("kode/data/batch_sangrai.csv").set_index("id_batch")

def hampel(x, w=7, t=6.0, minimal=5.0):                         #<1>
    """Tandai titik yang jauh dari median jendela (dalam MAD)."""
    x = pd.Series(x)
    med = x.rolling(w, center=True, min_periods=1).median()
    mad = (x - med).abs().rolling(w, center=True,
                                  min_periods=1).median()
    batas = np.maximum(t * 1.4826 * mad, minimal)
    return ((x - med).abs() > batas).to_numpy()

def ror(x, w=30):                                                #<2>
    """Laju kenaikan suhu (derajat per menit), dihaluskan."""
    x = pd.Series(x).rolling(9, center=True, min_periods=1).median()
    return (x.diff(w) * 60 / w).to_numpy()

kurva = {k: g.suhu_biji.to_numpy() for k, g in s.groupby("id_batch")}
T = 480                                                          #<3>
acuan = np.array([ror(kurva[k])[200:T] for k in b.index[:30]
                  if b.catatan_operator[k] == "normal"
                  and not hampel(kurva[k]).any()])
mu, sd = acuan.mean(0), acuan.std(0)
print(f"profil acuan RoR dari {len(acuan)} batch fase I")
print(f"RoR rata-rata pada detik 300: {mu[100]:.1f} C/menit")

baris = []
for k, x in kurva.items():                                       #<4>
    lonjak = int(hampel(x).sum())
    zr = (ror(x)[200:T] - mu) / sd
    lama = int((np.abs(zr) > 3).sum())                 # detik
    baris.append((k, lonjak, lama, b.catatan_operator[k]))
h = pd.DataFrame(baris, columns=["batch", "lonjakan", "detik_luar",
                                 "catatan"]).set_index("batch")
print("\nbatch bersinyal (ada lonjakan, atau RoR di luar pita")
print("lebih dari 30 detik):")
print(h[(h.lonjakan > 0) | (h.detik_luar > 30)].to_string())
lain = h.drop(["B017", "B071", "B038", "B055"])
print(f"\n76 batch lain: lonjakan maks {lain.lonjakan.max()}, "
      f"detik luar pita maks {lain.detik_luar.max()}")

with open("gambar/bab22/ror.dat", "w") as f:
    f.write("t lo hi b038 b055 b001\n")
    r38, r55 = ror(kurva["B038"]), ror(kurva["B055"])
    r01 = ror(kurva["B001"])
    for i in range(200, T, 2):
        f.write(f"{i} {mu[i - 200] - 3 * sd[i - 200]:.2f} "
                f"{mu[i - 200] + 3 * sd[i - 200]:.2f} {r38[i]:.2f} "
                f"{r55[i] if i < len(r55) else float('nan'):.2f} "
                f"{r01[i]:.2f}\n")
with open("gambar/bab22/lonjak.dat", "w") as f:
    f.write("t x tanda\n")
    x = kurva["B017"]; m = hampel(x)
    for i in range(0, len(x), 1):
        f.write(f"{i} {x[i]:.1f} {int(m[i])}\n")
