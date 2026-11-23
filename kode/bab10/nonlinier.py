import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DATA = "kode/data/"
h = pd.read_csv(DATA + "penjualan_harian.csv", parse_dates=["tanggal"])
g = pd.read_csv(DATA + "gerai.csv", parse_dates=["tanggal_buka"])
baru = g[g["tanggal_buka"].between("2023-04-01", "2024-03-31")]   #<1>
h = h.merge(baru[["id_gerai", "tanggal_buka"]], on="id_gerai")
h = h[(h["tanggal"] < "2026-01-01") & (h["tutup_lebaran"] == 0)]
h["bulan"] = (h["tanggal"] - h["tanggal_buka"]).dt.days // 30.44
bln = h.groupby(["id_gerai", "bulan"])["omzet"].mean().reset_index()
acuan = bln[bln["bulan"].between(13, 20)].groupby("id_gerai")["omzet"].mean()
bln["relatif"] = bln["omzet"] / bln["id_gerai"].map(acuan)       #<2>
bln = bln[bln["bulan"] <= 23]
t, r = bln["bulan"].to_numpy(), bln["relatif"].to_numpy()


def eksponen(t, a, tau):                                         #<3>
    return 1 - a * np.exp(-t / tau)


def patah(t, a, t0):                                             #<4>
    return np.where(t < t0, 1 - a * (1 - t / t0), 1.0)


pe, _ = curve_fit(eksponen, t, r, p0=[0.5, 6])
pp, _ = curve_fit(patah, t, r, p0=[0.5, 12])
p1 = np.polyfit(t, r, 2)                                         #<5>
for nama, yhat, k in [("polinom kuadrat", np.polyval(p1, t), 3),
                      ("eksponen jenuh", eksponen(t, *pe), 2),
                      ("garis patah", patah(t, *pp), 2)]:
    jkg = ((r - yhat) ** 2).sum()
    print(f"{nama:16s} JKG {jkg:.3f}  RMSE {np.sqrt(jkg / len(r)):.3f}")
print(f"eksponen: a = {pe[0]:.3f}, tau = {pe[1]:.2f} bulan")
print(f"garis patah: a = {pp[0]:.3f}, t0 = {pp[1]:.2f} bulan")
grid = np.linspace(0, 23, 47)
pd.DataFrame({"t": grid, "eks": eksponen(grid, *pe),
              "patah": patah(grid, *pp), "poli": np.polyval(p1, grid)}
             ).round(4).to_csv("gambar/bab10/ramp_kurva.dat", sep=" ",
                               index=False)
bln[["bulan", "relatif"]].round(4).to_csv("gambar/bab10/ramp_titik.dat",
                                          sep=" ", index=False)
