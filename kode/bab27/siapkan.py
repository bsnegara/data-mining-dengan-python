"""Modul bersama proyek retensi: atribut, rekayasa fitur, model."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "kode/data/"
NUMERIK = ["umur", "frekuensi_90h", "frekuensi_90h_lalu",
           "recency_hari", "nilai_90h", "rata_nilai_struk",
           "porsi_promo", "porsi_aplikasi", "porsi_kopi_hitam",
           "porsi_pagi", "porsi_akhir_pekan", "ragam_menu",
           "keluhan_90h", "jarak_km", "pakai_ulasan", "tren",
           "lama_anggota_bln"]
KATEGORIK = ["kota", "tingkat", "jenis_kelamin"]
NILAI_TAHAN, BIAYA = 180_000, 15_000   # Bab 7: nilai, biaya voucer


def fitur(df, potong, kategorik=KATEGORIK):
    """Semua atribut dihitung dari data sampai tanggal potong."""
    df = df.copy()
    df["tren"] = df.frekuensi_90h / (df.frekuensi_90h_lalu + 1)
    df["lama_anggota_bln"] = (pd.Timestamp(potong)
                              - pd.to_datetime(df.tanggal_daftar)
                              ).dt.days / 30.44
    return df[NUMERIK + kategorik]


def buat_model(C=3.162, kategorik=KATEGORIK):
    pra = ColumnTransformer(
        [("num", StandardScaler(), NUMERIK),
         ("kat", OneHotEncoder(handle_unknown="ignore"), kategorik)])
    return make_pipeline(pra, LogisticRegression(C=C, max_iter=3000))
