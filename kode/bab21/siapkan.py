"""Modul bantu Bab 21: matriks ketakmiripan dari penilaian panel."""
import numpy as np
import pandas as pd

def matriks_panel(kolom="skor_beda",
                  path="kode/data/kemiripan_panel.csv"):
    d = pd.read_csv(path)
    nama = list(dict.fromkeys(list(d.kopi_a) + list(d.kopi_b)))
    idx = {k: i for i, k in enumerate(nama)}
    D = np.zeros((len(nama), len(nama)))
    for _, r in d.iterrows():
        i, j = idx[r.kopi_a], idx[r.kopi_b]
        D[i, j] = D[j, i] = r[kolom]
    return D, nama
