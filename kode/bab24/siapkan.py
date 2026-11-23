import numpy as np
import pandas as pd

KEJADIAN = ["beranda", "promo", "cari", "lihat_menu", "detail_menu",
            "keranjang", "bayar", "selesai", "keluar"]
SELESAI, KELUAR = 7, 8


def muat(path="kode/data/sesi_aplikasi.csv", batas="2026-03-22"):
    d = pd.read_csv(path, parse_dates=["waktu"])
    d["kode"] = d.kejadian.map({k: i for i, k in enumerate(KEJADIAN)})
    g = d.sort_values(["id_sesi", "urutan"]).groupby("id_sesi")
    seq = g.kode.apply(np.array)
    info = pd.DataFrame({"mulai": g.waktu.min(),
                         "panjang": g.size(),
                         "beli": seq.map(lambda s: s[-1] == SELESAI)})
    info["latih"] = info.mulai < batas      # dibagi menurut waktu
    niat = g.niat_laten.apply(np.array)     # hanya di data sintetis
    return seq, info, niat


def pad(daftar):
    """Barisan beragam panjang -> matriks O dan topeng M."""
    T = max(map(len, daftar))
    O = np.zeros((len(daftar), T), int)
    M = np.zeros((len(daftar), T), bool)
    for i, s in enumerate(daftar):
        O[i, :len(s)], M[i, :len(s)] = s, True
    return O, M
