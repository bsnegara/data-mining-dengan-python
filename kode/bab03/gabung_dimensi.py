import pandas as pd

DATA = "kode/data/"
trx = pd.read_csv(DATA + "transaksi_gabungan_2026q1.csv",
                  parse_dates=["waktu"])
gerai = pd.read_csv(DATA + "gerai.csv")
menu = pd.read_csv(DATA + "menu.csv")

trx = trx.merge(                                          #<1>
    gerai[["id_gerai", "kota", "tipe_lokasi"]],
    left_on="kode_gerai", right_on="id_gerai",
    how="left", validate="many_to_one", indicator=True)
print(trx["_merge"].value_counts().to_string())           #<2>
trx = trx.drop(columns=["id_gerai", "_merge"])
trx = trx.merge(menu[["kode_menu", "nama_menu", "kategori"]],
                on="kode_menu", how="left",
                validate="many_to_one")

trx["nilai"] = trx["qty"] * trx["harga_satuan"] - trx["diskon"]  #<3>
print(trx[["no_struk", "kota", "nama_menu", "qty",
           "nilai"]].head(4).to_string())

per_struk = (trx.groupby("no_struk")                       #<4>
             .agg(item=("qty", "sum"), nilai=("nilai", "sum")))
print(per_struk.describe().round(0).to_string())
trx.to_csv(DATA + "transaksi_lengkap_2026q1.csv", index=False)
