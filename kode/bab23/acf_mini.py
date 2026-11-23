import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf, pacf

x = np.array([6.0, 6.4, 6.2, 6.8, 7.0, 6.6, 7.2, 7.4])           #<1>
d = x - x.mean()
r = [(d[k:] * d[:len(d) - k]).sum() / (d ** 2).sum() for k in (1, 2)]
print(f"rata-rata {x.mean():.2f}; r1 = {r[0]:.3f}, r2 = {r[1]:.3f}")
print("statsmodels acf:", acf(x, nlags=2).round(3))

mu, phi = x.mean(), 0.6                                          #<2>
ramal, akhir = [], x[-1]
for h in range(1, 4):
    akhir = mu + phi * (akhir - mu)
    ramal.append(round(float(akhir), 3))
print("ramalan AR(1) 3 langkah:", ramal)

print("\nACF dan PACF teoretis (lag 1-4):")                      #<3>
for nama, ar, ma in [("AR(1) phi=0,7", [1, -.7], [1]),
                     ("MA(1) theta=0,7", [1], [1, .7]),
                     ("AR(2) 0,5; 0,3", [1, -.5, -.3], [1])]:
    p = ArmaProcess(ar, ma)
    print(f"  {nama:16s} ACF  {p.acf(5)[1:].round(2)}")
    print(f"  {'':16s} PACF {p.pacf(5)[1:].round(2) + 0.0}")
