import numpy as np
import pywt

x = np.array([0, 2, 0, 2, 6, 8, 6, 8], float)                    #<1>
a, rinci = x.copy(), []
while len(a) > 1:
    rata = (a[0::2] + a[1::2]) / 2
    beda = (a[0::2] - a[1::2]) / 2
    rinci.append(beda)
    print(f"tingkat {len(rinci)}: rata-rata {rata}, rincian {beda}")
    a = rata
print("koefisien Haar:", a, *rinci[::-1])

def balik(a, rinci):                                             #<2>
    for d in rinci[::-1]:
        y = np.empty(2 * len(a))
        y[0::2], y[1::2] = a + d, a - d
        a = y
    return a

print("rekonstruksi  :", balik(a, rinci))
nol = [np.zeros_like(rinci[0])] + rinci[1:]                      #<3>
print("tanpa rincian tingkat 1:", balik(a, nol))

c = pywt.wavedec(x, "haar")                                      #<4>
print("\nPyWavelets (ortonormal):")
for nama, v in zip(["cA3", "cD3", "cD2", "cD1"], c):
    print(f"  {nama}: {np.round(v, 3)}")
skala = [2 ** 1.5, 2 ** 1.5, 2, np.sqrt(2)]
print("dibagi 2^(j/2):", [np.round(v / s, 3).tolist()
                         for v, s in zip(c, skala)])
print("energi sama:", np.sum(x ** 2),
      round(sum((v ** 2).sum() for v in c), 6))
