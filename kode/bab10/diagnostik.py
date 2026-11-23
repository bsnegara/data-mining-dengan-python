import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from gerai_tahunan import matang

rumus = "jumlah_kursi + luas_m2 + C(tipe_lokasi) + kota_besar"
lin = smf.ols("omzet ~ " + rumus, matang).fit()
log = smf.ols("np.log(omzet) ~ " + rumus, matang).fit()          #<1>
kota = smf.ols("np.log(omzet) ~ jumlah_kursi + luas_m2"          #<2>
               " + C(tipe_lokasi) + C(kota)", matang).fit()
for nama, m in [("linier", lin), ("log", log), ("log + kota", kota)]:
    print(f"{nama:10s}: R2 terkoreksi {m.rsquared_adj:.3f}, "
          f"AIC {m.aic:7.1f}")

X = lin.model.exog                                               #<3>
for j, nama in enumerate(lin.model.exog_names):
    if j > 0:
        print(f"VIF {nama[:28]:28s} {variance_inflation_factor(X, j):.2f}")

pengaruh = kota.get_influence()                                   #<4>
cook = pengaruh.cooks_distance[0]
i = np.argsort(-cook)[:3]
print("Cook terbesar:", ", ".join(f"{matang['id_gerai'].iloc[k]} "
                                  f"{cook[k]:.2f}" for k in i))
pd.DataFrame({"pas": np.r_[lin.fittedvalues],
              "sisa": np.r_[lin.resid],
              "pas_log": np.r_[log.fittedvalues],
              "sisa_log": np.r_[log.resid]}).round(4).to_csv(
    "gambar/bab10/sisa_gerai.dat", sep=" ", index=False)
print("sisa log + kota: maks", f"{np.abs(kota.resid).max():.3f}")
