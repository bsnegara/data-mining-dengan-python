import numpy as np
import pandas as pd
from halus import halus_median, halus_wavelet_ti, ror

s = pd.read_csv("kode/data/sensor_sangrai.csv")
b = pd.read_csv("kode/data/batch_sangrai.csv").set_index("id_batch")
kurva = {k: g.suhu_biji.to_numpy() for k, g in s.groupby("id_batch")}


def lonjakan(x, k=10):                                           #<1>
    """Rincian Haar tingkat 1 tak tercuplik: (x_t - x_t-1)/akar 2."""
    d = np.diff(x) / np.sqrt(2)
    sigma = np.median(np.abs(d)) / 0.6745
    tepi = np.where(np.abs(d) > k * sigma)[0]
    episode = []                  # tepi yang berjarak <= 3 detik
    for t in tepi:                # dianggap satu episode lonjakan
        if episode and t - episode[-1][-1] <= 3:
            episode[-1].append(t)
        else:
            episode.append([t])
    titik = [i for e in episode for i in range(e[0] + 1, e[-1] + 1)]
    return titik, len(episode), sigma


hasil, bersih = [], {}
for k, x in kurva.items():                                       #<2>
    idx, n_lonjak, sigma = lonjakan(x)
    y = x.astype(float).copy()
    y[idx] = np.nan
    y = pd.Series(y).interpolate().to_numpy()
    z = halus_wavelet_ti(y, mode="hard")
    bersih[k] = z
    balik = int(np.argmin(z[:200]))                              #<3>
    lewat = z[150:] > 196
    crack = int(np.argmax(lewat)) + 150 if lewat.any() else np.nan
    hasil.append({"batch": k, "lonjak": n_lonjak,
                  "sigma": round(sigma, 2), "balik": balik,
                  "suhu_balik": round(z[balik], 1), "crack": crack,
                  "crack_log": b.detik_crack[k],
                  "DTR": (len(x) - 1 - crack) / (len(x) - 1),
                  "catatan": b.catatan_operator[k]})
h = pd.DataFrame(hasil).set_index("batch")
h["crack"] = h.crack.astype("Int64")
print("sigma derau sensor: median", h.sigma.median(),
      " maksimum", h.sigma.max(), f"({h.sigma.idxmax()})")
print("lonjakan terdeteksi:", h.lonjak[h.lonjak > 0].to_dict())

print("\nmetrik pascabatch (kurva bersih):")                     #<4>
pilih = ["B001", "B002", "B003", "B017", "B038", "B055"]
print(h.loc[pilih, ["balik", "suhu_balik", "crack", "crack_log",
                    "DTR", "catatan"]].round(3).to_string())
selisih = (h.crack - h.crack_log).dropna()
jauh = list(selisih.index[selisih.abs() > 30])
print(f"crack bersih - crack log: median {selisih.median():.0f} "
      f"detik; beda > 30 detik: {jauh}")
normal = h[h.catatan == "normal"]
print(f"DTR batch normal: {normal.DTR.mean():.3f} +- "
      f"{normal.DTR.std():.3f}")

acuan = [k for k in b.index[:30] if h.catatan[k] == "normal"     #<5>
         and h.lonjak[k] == 0]
for nama, f in [("median 9 + selisih 30 s (Bab 22)",
                 lambda k: ror(halus_median(kurva[k]), 30)),
                ("wavelet + selisih 10 s",
                 lambda k: ror(bersih[k], 10))]:
    R = np.array([f(k)[200:480] for k in acuan])
    mu, sd = R.mean(0), R.std(0)
    luar = np.abs((f("B038")[200:480] - mu) / sd) > 3
    run = np.convolve(luar, np.ones(5), "valid") == 5
    print(f"{nama}: B038 bersinyal pada detik "
          f"{200 + np.argmax(run) + 4}")
pd.DataFrame({"t": np.arange(200, 480),
              "med": ror(halus_median(kurva["B038"]), 30)[200:480],
              "wav": ror(bersih["B038"], 10)[200:480],
              "mu": mu, "sd": sd}).to_csv(
    "gambar/bab26/ror.dat", sep=" ", index=False, float_format="%.3f")
