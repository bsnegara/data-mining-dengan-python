"""Modul bantu Bab 18: matriks utilitas dan atribut konten menu."""
import numpy as np
import pandas as pd

DATA = "kode/data/"
# atribut konten menu: bersusu, manis, dingin, kopi, makanan
KONTEN = {"K01": (1, 1, 1, 1, 0), "K02": (1, 1, 1, 1, 0),
          "K03": (1, 1, 1, 1, 0), "K04": (1, 0, 0, 1, 0),
          "K05": (1, 0, 0, 1, 0), "K06": (1, 1, 1, 1, 0),
          "H01": (0, 0, 0, 1, 0), "H02": (0, 0, 0, 1, 0),
          "H03": (0, 0, 0, 1, 0), "H04": (0, 0, 0, 1, 0),
          "H05": (0, 0, 0, 1, 0), "N01": (1, 1, 0, 0, 0),
          "N02": (1, 1, 0, 0, 0), "N03": (1, 1, 1, 0, 0),
          "N04": (0, 1, 1, 0, 0), "N05": (0, 1, 1, 0, 0),
          "M01": (0, 1, 0, 0, 1), "M02": (0, 1, 0, 0, 1),
          "M03": (0, 0, 0, 0, 1), "M04": (0, 0, 0, 0, 1),
          "M05": (0, 1, 0, 0, 1), "M06": (0, 1, 0, 0, 1)}


def muat(batas="2025-10-01"):
    r = pd.read_csv(DATA + "riwayat_aplikasi_2025.csv",
                    parse_dates=["tanggal"])
    menu = pd.read_csv(DATA + "menu.csv").set_index("kode_menu")
    kode = list(menu.index)
    lat, uji = r[r["tanggal"] < batas], r[r["tanggal"] >= batas]
    R = pd.crosstab(lat["id_anggota"], lat["kode_menu"]).reindex(
        columns=kode, fill_value=0)
    T = pd.crosstab(uji["id_anggota"], uji["kode_menu"]).reindex(
        columns=kode, fill_value=0)
    return R, T, menu


def konten(kode):
    return np.array([KONTEN[k] for k in kode], float)
