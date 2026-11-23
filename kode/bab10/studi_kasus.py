import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from gerai_tahunan import matang

rumus = {"linier": "omzet ~ luas_m2 + jumlah_kursi + C(tipe_lokasi)"
                   " + kota_besar",
         "log": "np.log(omzet) ~ luas_m2 + jumlah_kursi"
                " + C(tipe_lokasi) + kota_besar",
         "log+kota": "np.log(omzet) ~ luas_m2 + jumlah_kursi"
                     " + C(tipe_lokasi) + C(kota)"}
for nama, r in rumus.items():                                    #<1>
    galat = []
    for i in range(len(matang)):
        latih = matang.drop(matang.index[i])
        p = smf.ols(r, latih).fit().predict(matang.iloc[[i]]).iloc[0]
        p = np.exp(p) if nama.startswith("log") else p
        galat.append(p / matang["omzet"].iloc[i] - 1)
    galat = np.abs(galat)
    print(f"{nama:8s} LOOCV: MAPE {galat.mean():.1%}, "
          f"galat terbesar {galat.max():.1%}")

m = smf.ols(rumus["log+kota"], matang).fit()
print("\nefek (persen):")                                        #<2>
for k, v in m.params.items():
    if k != "Intercept":
        k2 = k.replace("C(tipe_lokasi)[T.", "tipe ").replace(
            "C(kota)[T.", "kota ").replace("]", "")
        print(f"  {k2:22s}"
              f" {100 * (np.exp(v) - 1):+6.1f}%  (p = {m.pvalues[k]:.3f})")

calon = pd.DataFrame({                                           #<3>
    "lokasi": ["Pekanbaru, kampus", "Jakarta, perkantoran",
               "Medan, mal", "Padang, permukiman", "Batam, mal"],
    "luas_m2": [80, 70, 95, 110, 60],
    "jumlah_kursi": [40, 30, 35, 45, 28],
    "tipe_lokasi": ["kampus", "perkantoran", "mal", "permukiman", "mal"],
    "kota": ["Pekanbaru", "Jakarta", "Medan", "Padang", "Batam"]})
pr = m.get_prediction(calon).summary_frame(alpha=0.10)           #<4>
calon["ramalan"] = np.exp(pr["mean"]).round(0)
calon["bawah90"] = np.exp(pr["obs_ci_lower"]).round(0)
calon["atas90"] = np.exp(pr["obs_ci_upper"]).round(0)
calon["thn1_juta"] = (calon["ramalan"] * 365 * 0.75 / 1000).round(0)  #<5>
print("\nramalan omzet harian matang (ribu Rp), selang 90%:")
print(calon[["lokasi", "ramalan", "bawah90", "atas90",
             "thn1_juta"]].to_string(index=False))
