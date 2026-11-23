"""Modul bantu Bab 13: tensor untuk jaringan PyTorch belanja bulan depan."""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "kode/data/"
NUMERIK = ["umur", "frekuensi_90h", "frekuensi_90h_lalu", "recency_hari",
           "nilai_90h", "rata_nilai_struk", "porsi_promo", "porsi_aplikasi",
           "porsi_kopi_hitam", "porsi_pagi", "porsi_akhir_pekan",
           "ragam_menu", "keluhan_90h", "jarak_km", "pakai_ulasan"]
KATEGORIK = ["kota", "tingkat", "jenis_kelamin"]


def data_tensor(seed=7):
    df = pd.read_csv(DATA + "pelanggan.csv")
    df["gerai"] = df["gerai_utama"].str[1:].astype(int) - 1      # 0..39
    y = (df["belanja_bulan_depan"] / 1000).to_numpy(np.float32)
    latih, uji = train_test_split(np.arange(len(df)), test_size=0.3,
                                  random_state=seed)
    sk = StandardScaler().fit(df.loc[latih, NUMERIK])
    oh = OneHotEncoder(sparse_output=False).fit(df.loc[latih, KATEGORIK])
    Xn = np.hstack([sk.transform(df[NUMERIK]),
                    oh.transform(df[KATEGORIK])]).astype(np.float32)
    t = lambda a: torch.tensor(a)
    return (df, t(Xn), torch.tensor(df["gerai"].to_numpy()), t(y),
            latih, uji)
