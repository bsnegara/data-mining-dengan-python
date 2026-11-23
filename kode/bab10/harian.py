import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA = "kode/data/"
h = pd.read_csv(DATA + "penjualan_harian.csv", parse_dates=["tanggal"])
g = pd.read_csv(DATA + "gerai.csv", parse_dates=["tanggal_buka"])
h = h.merge(g[["id_gerai", "tipe_lokasi", "tanggal_buka"]], on="id_gerai")
h = h[(h["tanggal"] < "2026-01-01") & (h["tutup_lebaran"] == 0)
      & (h["tanggal"] - h["tanggal_buka"] > pd.Timedelta(days=365))]  #<1>
h["hari"] = h["tanggal"].dt.dayofweek
h["gajian"] = ((h["tanggal"].dt.day >= 25)
               | (h["tanggal"].dt.day <= 3)).astype(int)
h["tahun"] = h["tanggal"].dt.year - 2023
h["libur_kampus"] = ((h["tipe_lokasi"] == "kampus")
                     & (h["tanggal"].dt.month.isin([7, 8]))).astype(int)
rumus = ("np.log(omzet) ~ C(id_gerai) + C(hari) + tahun + gajian"  #<2>
         " + hujan_mm + libur_kampus + libur_nasional"
         " + libur_nasional:C(tipe_lokasi) + ramadan"
         " + ramadan:C(tipe_lokasi)")
m = smf.ols(rumus, h).fit()
print(f"{len(h)} hari-gerai, {len(m.params)} koefisien, "
      f"R2 = {m.rsquared:.3f}")
efek = {"Sabtu vs Senin": "C(hari)[T.5]", "per tahun": "tahun",
        "gajian": "gajian", "hujan per 10 mm": "hujan_mm",
        "libur kampus (Jul-Agu)": "libur_kampus",
        "libur nasional, kampus": "libur_nasional",
        "Ramadan, kampus": "ramadan",
        "Ramadan, mal (tambahan)": "ramadan:C(tipe_lokasi)[T.mal]"}
for nama, k in efek.items():                                     #<3>
    b = m.params[k] * (10 if "hujan" in nama else 1)
    print(f"{nama:24s} {100 * (np.exp(b) - 1):+6.1f}%")

lin = smf.ols(rumus.replace("np.log(omzet)", "omzet"), h).fit()  #<4>
for nama, mm, pas in [("linier", lin, lin.fittedvalues),
                      ("log", m, np.exp(m.fittedvalues))]:
    kel = pd.qcut(pas, 5, labels=False)
    sd = mm.resid.groupby(kel).std() / mm.resid.std()
    print(f"sd sisa per kuintil ramalan, {nama:6s}:",
          " ".join(f"{v:.2f}" for v in sd))
sampel = h.sample(1500, random_state=0).index
pd.DataFrame({"pas": lin.fittedvalues[sampel] / 1000,
              "sisa": lin.resid[sampel] / 1000,
              "pas_log": m.fittedvalues[sampel],
              "sisa_log": m.resid[sampel]}).round(4).to_csv(
    "gambar/bab10/sisa_harian.dat", sep=" ", index=False)
