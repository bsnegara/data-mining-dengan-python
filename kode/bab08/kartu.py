"""Modul bantu Bab 8: atribut anggota yang didiskretkan untuk kartu skor."""
import numpy as np
import pandas as pd

DATA = "kode/data/"
df = pd.read_csv(DATA + "pelanggan.csv")
df["tren"] = df["frekuensi_90h"] / (df["frekuensi_90h_lalu"] + 1)  #<1>
y = df["churn"]

BATAS = {"recency_hari": [7, 14, 30, 45],                         #<2>
         "tren": [0.5, 0.8, 1.1],
         "keluhan_90h": [0.5, 1.5],
         "ragam_menu": [2.5, 4.5, 6.5],
         "porsi_promo": [0.2, 0.4],
         "pakai_ulasan": [0.5]}
kartu = pd.DataFrame({k: np.digitize(df[k], b) for k, b in BATAS.items()})
kartu["tingkat"] = df["tingkat"].map({"perunggu": 0, "perak": 1,
                                      "emas": 2})
