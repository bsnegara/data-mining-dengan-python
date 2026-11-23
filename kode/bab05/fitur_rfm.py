import pandas as pd

DATA = "kode/data/"
trx = pd.read_csv(DATA + "transaksi_lengkap_2026q1.csv",
                  usecols=["no_struk", "waktu", "id_anggota", "nilai"],
                  parse_dates=["waktu"])
trx = trx.dropna(subset=["id_anggota"])
potong = pd.Timestamp("2026-04-01")                              #<1>

struk = (trx.groupby(["id_anggota", "no_struk"])                 #<2>
         .agg(waktu=("waktu", "max"), nilai=("nilai", "sum"))
         .reset_index())
rfm = struk.groupby("id_anggota").agg(                           #<3>
    recency=("waktu", lambda w: (potong - w.max()).days),
    frequency=("no_struk", "nunique"),
    monetary=("nilai", "sum"))
print(rfm.shape)
print(rfm.describe().loc[["mean", "50%", "max"]].round(0).to_string())

skor = pd.DataFrame({                                            #<4>
    "R": pd.qcut(rfm["recency"].rank(method="first"), 5,
                 labels=[5, 4, 3, 2, 1]),
    "F": pd.qcut(rfm["frequency"].rank(method="first"), 5,
                 labels=[1, 2, 3, 4, 5]),
    "M": pd.qcut(rfm["monetary"].rank(method="first"), 5,
                 labels=[1, 2, 3, 4, 5])}).astype(int)
rfm["segmen"] = skor.astype(str).sum(axis=1)
print(rfm["segmen"].value_counts().head(4).to_string())
print(rfm.loc[rfm["segmen"] == "555"].head(2).to_string())
