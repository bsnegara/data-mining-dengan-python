import sys; sys.path.insert(0, "kode/bab23")
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from siapkan import mingguan
warnings.simplefilter("ignore")

w = mingguan()
t = np.arange(len(w))
for k in (1, 2):                                                 #<1>
    w[f"sin{k}"] = np.sin(2 * np.pi * k * t / 52.18)
    w[f"cos{k}"] = np.cos(2 * np.pi * k * t / 52.18)
FOURIER = ["sin1", "cos1", "sin2", "cos2"]
KALENDER = ["ramadan", "tutup", "libur", "pasca"]                #<2>

def ramal(latih, uji):                                           #<3>
    y = latih.per_gerai
    h = len(uji)
    out = {"naif musiman": w.per_gerai.shift(52).loc[uji.index]}
    hw = ExponentialSmoothing(y, trend="add", damped_trend=True,
                              seasonal="add",
                              seasonal_periods=52).fit()
    out["Holt-Winters"] = pd.Series(hw.forecast(h).to_numpy(),
                                    uji.index)
    for nama, X in [("ARIMA + Fourier", FOURIER),
                    ("ARIMA + Fourier + kalender", FOURIER + KALENDER)]:
        m = SARIMAX(y, exog=latih[X], order=(1, 1, 1))
        m = m.fit(disp=False)
        f = m.get_forecast(h, exog=uji[X])
        out[nama] = pd.Series(f.predicted_mean.to_numpy(), uji.index)
        if "kalender" in nama:
            pi = {a: f.conf_int(alpha=a).to_numpy()
                  for a in (.2, .05)}
            koef = m.params[KALENDER]
    return out, pi, koef

UJI = {"Ramadan 2025": ("2024-12-29", "2025-04-13"),             #<4>
       "Ramadan 2026": ("2025-12-28", "2026-03-29")}
for judul, (akhir, ujung) in UJI.items():
    latih = w[:akhir]
    uji = w[pd.Timestamp(akhir) + pd.Timedelta(days=7):ujung]
    out, pi, koef = ramal(latih, uji)
    skala = latih.per_gerai.diff().abs().mean()        # MASE
    print(f"\n{judul}: latih s.d. {akhir}, uji {len(uji)} minggu")
    print(f"  {'model':28s} {'MAE':>6s} {'MAPE':>6s} {'MASE':>5s}"
          f" {'MAE Ramadan':>11s}")
    ram = (uji.ramadan > 0) | (uji.tutup > 0)
    for nama, f in out.items():                                  #<5>
        e = (uji.per_gerai - f).abs()
        print(f"  {nama:28s} {e.mean():6.3f} "
              f"{(e / uji.per_gerai).mean():6.1%} "
              f"{e.mean() / skala:5.2f} {e[ram].mean():11.3f}")
    for a, (nm) in [(.2, "80%"), (.05, "95%")]:                  #<6>
        lo, hi = pi[a][:, 0], pi[a][:, 1]
        cak = ((uji.per_gerai >= lo) & (uji.per_gerai <= hi)).mean()
        print(f"  cakupan selang {nm}: {cak:.0%}")
print("\nkoefisien kalender (kg per gerai per minggu, per hari):")
print("  " + ", ".join(f"{k} {v:+.3f}" for k, v in koef.items()))

f = out["ARIMA + Fourier + kalender"]                            #<7>
total = pd.DataFrame({"aktual": uji.biji, "ramalan": f * uji.gerai,
                      "atas90": pi[.2][:, 1] * uji.gerai})
print("\nkebutuhan total 2026 (kg), contoh minggu Ramadan:")
print(total.loc["2026-02-15":"2026-03-29"].round(0).to_string())
with open("gambar/bab23/ramalan.dat", "w") as fh:
    fh.write("t y f lo80 hi80 lo95 hi95 naif\n")
    for i, d in enumerate(uji.index):
        fh.write(f"{i + 1} {uji.per_gerai[d]:.3f} "
                 f"{f[d]:.3f} {pi[.2][i, 0]:.3f} {pi[.2][i, 1]:.3f} "
                 f"{pi[.05][i, 0]:.3f} {pi[.05][i, 1]:.3f} "
                 f"{out['naif musiman'][d]:.3f}\n")
with open("gambar/bab23/latih.dat", "w") as fh:
    fh.write("t y\n")
    for d, v in w.per_gerai["2025-06-01":"2025-12-28"].items():
        fh.write(f"{(d - uji.index[0]).days // 7 + 1} {v:.3f}\n")
