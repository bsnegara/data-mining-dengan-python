import pandas as pd

DATA = "kode/data/"
pel = pd.read_csv(DATA + "pelanggan.csv", parse_dates=["tanggal_daftar"])
pel = pel.drop(columns=["segmen_laten", "belanja_bulan_depan"])   #<1>
print(pel.shape)
print(pel.dtypes.value_counts().to_string())

num = ["umur", "frekuensi_90h", "recency_hari", "nilai_90h"]
ring = pel[num].describe().T                                     #<2>
ring = ring[["mean", "std", "min", "50%", "max"]]
ring.loc["nilai_90h"] /= 1000            # dalam ribu rupiah
print(ring.round(1).to_string())

print(pel["tingkat"].value_counts().to_string())                 #<3>
print(pel["kota"].value_counts(normalize=True)
      .mul(100).round(1).head(4).to_string())
print(pel["kota"].mode()[0], pel["nilai_90h"].mode()[0])          #<4>
