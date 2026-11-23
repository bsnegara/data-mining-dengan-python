"""Modul bantu Bab 14: atribut laboratorium biji kopi (tanpa skor cupping)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "kode/data/"
LAB = ["ketinggian_mdpl", "lama_simpan_hari", "kadar_air", "cacat_primer",
       "cacat_sekunder", "ukuran_screen", "densitas_gl", "jamur"]
KAT = ["asal", "varietas", "proses", "musim_panen", "cara_jemur"]


def muat():
    df = pd.read_csv(DATA + "mutu_biji.csv")
    return df, df[LAB + KAT], df["specialty"]


def pra():
    return ColumnTransformer([("num", StandardScaler(), LAB),
                              ("kat", OneHotEncoder(), KAT)])
