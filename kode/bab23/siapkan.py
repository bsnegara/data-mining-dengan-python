"""Modul bantu Bab 23: deret mingguan kebutuhan biji dan kalender."""
import pandas as pd

LEBARAN = ["2023-04-22", "2024-04-10", "2025-03-31", "2026-03-20"]


def harian():
    d = pd.read_csv("kode/data/penjualan_harian.csv",
                    parse_dates=["tanggal"])
    h = d.groupby("tanggal").agg(
        biji=("biji_kg", "sum"),
        gerai=("id_gerai", "nunique"),    # gerai yang sudah buka
        ramadan=("ramadan", "max"),
        tutup=("tutup_lebaran", "max"),
        libur=("libur_nasional", "max"))
    pasca = pd.Series(0, index=h.index)
    for t in pd.to_datetime(LEBARAN):
        pasca[(h.index > t + pd.Timedelta(days=1))
              & (h.index <= t + pd.Timedelta(days=8))] = 1
    h["pasca"] = pasca
    return h


def mingguan():
    """Minggu Senin-Minggu yang lengkap; kebutuhan biji per gerai."""
    h = harian()
    w = h.resample("W-SUN").agg({"biji": "sum", "gerai": "mean",
                                 "ramadan": "sum", "tutup": "sum",
                                 "libur": "sum", "pasca": "sum"})
    w = w[h.resample("W-SUN").size() == 7]
    w["per_gerai"] = w.biji / w.gerai
    return w
