import sys; sys.path.insert(0, "kode/bab23")
import warnings
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, pacf
from siapkan import mingguan
warnings.simplefilter("ignore")

w = mingguan()
y = w.per_gerai[:"2025-12-28"]                                   #<1>
for nama, s in [("level", y), ("selisih pertama", y.diff().dropna())]:
    stat, p = adfuller(s, autolag="AIC")[:2]                     #<2>
    print(f"ADF {nama:16s}: statistik {stat:6.2f}, nilai-p {p:.3f}")

dy = y.diff().dropna()                                           #<3>
a, pa = acf(dy, nlags=8), pacf(dy, nlags=8)
batas = 1.96 / np.sqrt(len(dy))
print(f"\nACF/PACF selisih pertama (batas +-{batas:.2f}):")
print("lag   " + " ".join(f"{k:6d}" for k in range(1, 9)))
print("ACF   " + " ".join(f"{v:6.2f}" for v in a[1:]))
print("PACF  " + " ".join(f"{v:6.2f}" for v in pa[1:]))

print("\nkandidat ARIMA(p,1,q), diurutkan menurut AIC:")         #<4>
hasil = []
for p in range(3):
    for q in range(3):
        m = ARIMA(y, order=(p, 1, q)).fit()
        hasil.append((m.aic, p, q, m))
for aic, p, q, m in sorted(hasil, key=lambda r: r[0])[:4]:
    lb = acorr_ljungbox(m.resid[1:], lags=[10]).lb_pvalue.iloc[0]  #<5>
    print(f"  ARIMA({p},1,{q})  AIC {aic:7.1f}  Ljung-Box p {lb:.3f}")
with open("gambar/bab23/acf.dat", "w") as f:
    f.write("lag acf pacf\n")
    a, pa = acf(dy, nlags=20), pacf(dy, nlags=20)
    for k in range(1, 21):
        f.write(f"{k} {a[k]:.3f} {pa[k]:.3f}\n")
