import sys
import numpy as np
import pandas as pd
import pywt
from halus import halus_rata, halus_wavelet_ti
sys.path.insert(0, "kode/data")
from bangkitkan_data_lanjut import kurva_sangrai

s = pd.read_csv("kode/data/sensor_sangrai.csv")
x = s[s.id_batch == "B017"].suhu_biji.to_numpy()
c = pywt.wavedec(x, "sym8", level=5, mode="smooth")


def bagian(j):
    """Sumbangan satu tingkat koefisien ke sinyal."""
    z = [np.zeros_like(v) for v in c]
    z[j] = c[j]
    return pywt.waverec(z, "sym8", mode="smooth")[:len(x)]


pd.DataFrame({"t": np.arange(len(x)), "x": x, "a5": bagian(0),
              "d4": bagian(2), "d2": bagian(4),
              "d1": bagian(5)}).to_csv(
    "gambar/bab26/dekomposisi.dat", sep=" ", index=False,
    float_format="%.3f")

rng = np.random.default_rng(7)
t, benar, _ = kurva_sangrai(rng, 690, 210)
y = benar + rng.normal(0, .7, len(t))
m = (t >= 40) & (t <= 160)
pd.DataFrame({"t": t, "benar": benar, "y": y, "rata": halus_rata(y),
              "wav": halus_wavelet_ti(y, mode="hard")})[m].to_csv(
    "gambar/bab26/halus.dat", sep=" ", index=False,
    float_format="%.3f")
