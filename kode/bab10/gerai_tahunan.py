"""Modul bantu Bab 10: satu baris per gerai, omzet harian rata-rata 2025."""
import pandas as pd

DATA = "kode/data/"
harian = pd.read_csv(DATA + "penjualan_harian.csv", parse_dates=["tanggal"])
gerai = pd.read_csv(DATA + "gerai.csv", parse_dates=["tanggal_buka"])
omzet = (harian[harian["tanggal"].dt.year == 2025]
         .groupby("id_gerai")["omzet"].mean() / 1000).rename("omzet")
gerai = gerai.merge(omzet, left_on="id_gerai", right_index=True)
gerai["kota_besar"] = gerai["kota"].isin(
    ["Jakarta", "Bandung", "Medan"]).astype(int)
matang = gerai[gerai["tanggal_buka"] < "2024-01-01"].copy()      # 32 gerai
