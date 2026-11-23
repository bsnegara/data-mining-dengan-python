"""Modul bantu Bab 7: memuat data churn dan membuat pipeline standar."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "kode/data/"
NUMERIK = ["umur", "frekuensi_90h", "frekuensi_90h_lalu", "recency_hari",
           "nilai_90h", "rata_nilai_struk", "porsi_promo", "porsi_aplikasi",
           "porsi_kopi_hitam", "porsi_pagi", "porsi_akhir_pekan",
           "ragam_menu", "keluhan_90h", "jarak_km", "pakai_ulasan"]
KATEGORIK = ["kota", "tingkat", "jenis_kelamin"]


def muat():
    df = pd.read_csv(DATA + "pelanggan.csv")
    return df[NUMERIK + KATEGORIK], df["churn"]


def pipa(model, skala=True):
    num = StandardScaler() if skala else "passthrough"
    pra = ColumnTransformer([
        ("num", num, NUMERIK),
        ("kat", OneHotEncoder(handle_unknown="ignore"), KATEGORIK)])
    return make_pipeline(pra, model)
