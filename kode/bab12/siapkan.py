"""Modul bantu Bab 12: data churn dengan atribut tren dan praproses."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

DATA = "kode/data/"
NUMERIK = ["umur", "frekuensi_90h", "frekuensi_90h_lalu", "recency_hari",
           "nilai_90h", "rata_nilai_struk", "porsi_promo", "porsi_aplikasi",
           "porsi_kopi_hitam", "porsi_pagi", "porsi_akhir_pekan",
           "ragam_menu", "keluhan_90h", "jarak_km", "pakai_ulasan"]
KATEGORIK = ["kota", "tingkat", "jenis_kelamin"]


def muat(tren=False):
    df = pd.read_csv(DATA + "pelanggan.csv")
    num = list(NUMERIK)
    if tren:
        df["tren"] = df["frekuensi_90h"] / (df["frekuensi_90h_lalu"] + 1)
        num.append("tren")
    return df[num + KATEGORIK], df["churn"], num


def pra_pohon(num):
    """Pohon tidak butuh penskalaan; kategori cukup dikodekan ordinal."""
    return ColumnTransformer(
        [("num", "passthrough", num),
         ("kat", OrdinalEncoder(), KATEGORIK)],
        verbose_feature_names_out=False)


def pra_onehot(num):
    return ColumnTransformer(
        [("num", "passthrough", num),
         ("kat", OneHotEncoder(handle_unknown="ignore"), KATEGORIK)],
        verbose_feature_names_out=False)
