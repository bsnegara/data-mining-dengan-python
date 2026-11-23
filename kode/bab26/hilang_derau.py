import sys
from functools import partial
import numpy as np
import pywt
from halus import (halus_median, halus_rata, halus_wavelet,
                   halus_wavelet_ti, ror)
sys.path.insert(0, "kode/data")
from bangkitkan_data_lanjut import kurva_sangrai

def crack(x, t0=150, suhu=196):
    """Detik pertama suhu biji melewati 196 C setelah titik balik."""
    return int(np.argmax(x[t0:] > suhu)) + t0


rng = np.random.default_rng(2627)                                #<1>
metode = {"tanpa penghalusan": lambda x: x,
          "rata-rata bergerak 15": halus_rata,
          "median bergerak 9": halus_median,
          "wavelet sym8 lunak": halus_wavelet,
          "wavelet haar lunak": partial(halus_wavelet,
                                        wavelet="haar"),
          "wavelet sym8 keras": partial(halus_wavelet, mode="hard"),
          "wavelet keras + geser": partial(halus_wavelet_ti,
                                           mode="hard")}
galat = {m: [] for m in metode}
for _ in range(40):
    t, benar, _ = kurva_sangrai(rng, int(rng.normal(690, 25)),
                                rng.normal(210, 2.5))
    x = benar + rng.normal(0, 0.7, len(t))                       #<2>
    rb = ror(benar, 10)
    for m, f in metode.items():                                  #<3>
        h = f(x)
        balik = np.argmin(h[:200]) - np.argmin(benar[:200])
        galat[m].append([np.sqrt(np.mean((h - benar) ** 2)),
                         np.sqrt(np.nanmean((ror(h, 10) - rb) ** 2)),
                         abs(balik), crack(h) - crack(benar)])
print(f"{'metode':22s}   RMSE   RMSE  |galat|   galat")
print(f"{'':22s}   suhu    RoR    balik   crack")
for m, g in galat.items():
    g = np.array(g).mean(0)
    print(f"{m:22s} {g[0]:6.3f} {g[1]:6.2f} {g[2]:7.1f}s"
          f" {g[3]:6.1f}s")

c = pywt.wavedec(x, "sym8", level=5, mode="smooth")              #<4>
sigma = np.median(np.abs(c[-1])) / 0.6745
ambang = sigma * np.sqrt(2 * np.log(len(x)))
nol = sum((np.abs(d) <= ambang).sum() for d in c[1:])
print(f"\nsigma derau dari rincian tingkat 1: {sigma:.3f}"
      " (benar 0,7)")
print(f"ambang universal {ambang:.2f}; koefisien rincian dinolkan: "
      f"{nol} dari {sum(len(d) for d in c[1:])}")
print("panjang koefisien per tingkat:", [len(v) for v in c])
