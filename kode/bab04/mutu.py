import pandas as pd

DATA = "kode/data/"
mentah = pd.read_csv(DATA + "pelanggan_mentah.csv")
print(mentah.shape)
kosong = mentah.isna().mean().mul(100).round(1)                   #<1>
print(kosong[kosong > 0].to_string())
print("duplikat penuh:", mentah.duplicated().sum())               #<2>
print("duplikat id   :", mentah["id_anggota"].duplicated().sum())
import textwrap                                                   #<3>
print(textwrap.fill(", ".join(sorted(mentah["kota"].unique())), 60))
umur = pd.to_numeric(mentah["umur"], errors="coerce")
print("umur < 15 atau > 90:", int(((umur < 15) | (umur > 90)).sum()))  #<4>
teks = pd.to_numeric(mentah["nilai_90h"], errors="coerce").isna()
print("nilai_90h bukan angka:", int(teks.sum()))                  #<5>
print(mentah.loc[teks, "nilai_90h"].head(2).tolist())
