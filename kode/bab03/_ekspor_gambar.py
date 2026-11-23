"""Mengekspor data gambar Bab 3 untuk digambar ulang dengan pgfplots."""
import pandas as pd

DATA = "kode/data/"
trx = pd.read_csv(DATA + "transaksi_lengkap_2026q1.csv",
                  parse_dates=["waktu"])
harian = trx.set_index("waktu")["nilai"].resample("D").sum() / 1e6
pd.DataFrame({"hari": range(len(harian)), "omzet": harian.round(2).values}
             ).to_csv("gambar/bab03/omzet_harian.dat", sep=" ", index=False)
trx["hari"] = trx["waktu"].dt.dayofweek
trx["jam"] = trx["waktu"].dt.hour
biasa = trx[trx["waktu"] < "2026-02-18"]
peta = biasa.pivot_table(index="hari", columns="jam", values="no_struk",
                         aggfunc="nunique")
rows = [(j, h, int(peta.loc[h, j])) for h in peta.index for j in peta.columns]
with open("gambar/bab03/peta_jam_hari.dat", "w") as f:
    f.write("jam hari struk\n")
    last = None
    for j, h, v in rows:
        if last is not None and h != last:
            f.write("\n")
        f.write(f"{j} {h} {v}\n"); last = h
print(peta.values.min(), peta.values.max())
