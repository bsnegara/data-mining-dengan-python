import numpy as np
import pandas as pd

DATA = "kode/data/"
df = pd.read_csv(DATA + "pelanggan_mentah.csv")
print("awal          :", df.shape)

df = df.drop_duplicates()                                        #<1>
print("tanpa duplikat:", df.shape)

KOTA = {"pekanbaru": "Pekanbaru", "pekan baru": "Pekanbaru",     #<2>
        "jogja": "Yogyakarta", "yogya": "Yogyakarta",
        "diy": "Yogyakarta", "dki jakarta": "Jakarta",
        "bdg": "Bandung"}
kunci = df["kota"].str.strip().str.lower()
df["kota"] = kunci.map(KOTA).fillna(kunci.str.title())
print("kota:", df["kota"].nunique(), "nilai unik")

df["tanggal_daftar"] = pd.to_datetime(df["tanggal_daftar"],      #<3>
                                      format="mixed", dayfirst=True)

teks = df["nilai_90h"].astype(str)                               #<4>
df["nilai_90h"] = pd.to_numeric(
    teks.str.replace("Rp", "").str.replace(".", "", regex=False))

df["umur"] = pd.to_numeric(df["umur"])                           #<5>
salah = ~df["umur"].between(15, 90) & df["umur"].notna()
df.loc[salah, "umur"] = np.nan
print("umur tak wajar -> NaN:", int(salah.sum()))

meter = df["jarak_km"] > 100                                     #<6>
df.loc[meter, "jarak_km"] = df.loc[meter, "jarak_km"] / 1000
print("jarak dari meter:", int(meter.sum()))

df.to_csv(DATA + "pelanggan_bersih.csv", index=False)
acuan = pd.read_csv(DATA + "pelanggan.csv").set_index("id_anggota")
cek = df.set_index("id_anggota").loc[acuan.index]                #<7>
print("nilai_90h cocok :", (cek["nilai_90h"] == acuan["nilai_90h"]).all())
print("kota cocok      :", (cek["kota"] == acuan["kota"]).all())
