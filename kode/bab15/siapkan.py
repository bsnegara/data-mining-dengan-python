"""Modul bantu Bab 15: atribut perilaku anggota untuk segmentasi."""
import numpy as np
import pandas as pd

DATA = "kode/data/"
PERILAKU = ["frekuensi_90h", "rata_nilai_struk", "porsi_pagi",
            "porsi_akhir_pekan", "porsi_promo", "porsi_kopi_hitam",
            "porsi_aplikasi"]
MIRING = ["frekuensi_90h", "rata_nilai_struk"]


def muat(log=True):
    df = pd.read_csv(DATA + "pelanggan.csv")
    X = df[PERILAKU].copy()
    if log:
        X[MIRING] = np.log1p(X[MIRING])
    return df, X
