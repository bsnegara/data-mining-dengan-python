import pandas as pd

DATA = "kode/data/"
kasir = pd.read_csv(DATA + "transaksi_kasir_2026q1.csv")   #<1>
app = pd.read_csv(DATA + "pesanan_aplikasi_2026q1.csv")

print(kasir.shape, app.shape)                             #<2>
print(kasir.iloc[:3, :5].to_string())
print()
kasir.info()                                              #<3>
print()
print(kasir["metode_bayar"].value_counts())               #<4>
