"""Modul bantu Bab 16: indeks pola penjualan setiap gerai, 2023-2025."""
import pandas as pd

DATA = "kode/data/"
INDEKS = ["sabtu", "minggu", "libur_nasional", "libur_semester", "ramadan",
          "hujan"]


def indeks_gerai():
    d = pd.read_csv(DATA + "penjualan_harian.csv", parse_dates=["tanggal"])
    d = d[d["tutup_lebaran"] == 0]
    d["hari"] = d["tanggal"].dt.dayofweek
    d["bulan"] = d["tanggal"].dt.month
    baris = {}
    for g, x in d.groupby("id_gerai"):
        biasa = x[(x["hari"] < 5) & (x["ramadan"] == 0)
                  & (x["libur_nasional"] == 0)]["omzet"].mean()
        lain = x[x["ramadan"] == 0]["omzet"].mean()
        baris[g] = {
            "sabtu": x[x["hari"] == 5]["omzet"].mean() / biasa,
            "minggu": x[x["hari"] == 6]["omzet"].mean() / biasa,
            "libur_nasional": x[x["libur_nasional"] == 1]["omzet"].mean()
            / biasa,
            "libur_semester": x[x["bulan"].isin([7, 8])]["omzet"].mean()
            / x[~x["bulan"].isin([7, 8])]["omzet"].mean(),
            "ramadan": x[x["ramadan"] == 1]["omzet"].mean() / lain,
            "hujan": x[x["hujan_mm"] > 20]["omzet"].mean()
            / x[x["hujan_mm"] == 0]["omzet"].mean()}
    gerai = pd.read_csv(DATA + "gerai.csv").set_index("id_gerai")
    return pd.DataFrame(baris).T[INDEKS], gerai
