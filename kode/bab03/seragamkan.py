import pandas as pd

DATA = "kode/data/"
kasir = pd.read_csv(DATA + "transaksi_kasir_2026q1.csv",
                    parse_dates=["waktu"])                #<1>
app = pd.read_csv(DATA + "pesanan_aplikasi_2026q1.csv")
gerai = pd.read_csv(DATA + "gerai.csv")
menu = pd.read_csv(DATA + "menu.csv")

app = app.rename(columns={                                #<2>
    "order_id": "no_struk", "created_at": "waktu",
    "quantity": "qty", "unit_price": "harga_satuan",
    "promo_amount": "diskon", "payment": "metode_bayar",
    "member_id": "id_anggota"})
app["waktu"] = (pd.to_datetime(app["waktu"])              #<3>
                .dt.tz_localize(None))
peta_gerai = dict(zip(gerai["kode_aplikasi"], gerai["id_gerai"]))
peta_menu = dict(zip(menu["nama_menu"], menu["kode_menu"]))
app["kode_gerai"] = app["outlet"].map(peta_gerai)         #<4>
app["kode_menu"] = app["item_name"].map(peta_menu)
app["metode_bayar"] = app["metode_bayar"].replace({
    "e-wallet": "Dompet Digital", "qris": "QRIS",
    "card": "Kartu Kredit", "virtual_account": "Transfer VA"})

kolom = ["no_struk", "waktu", "kode_gerai", "kode_menu", "qty",
         "harga_satuan", "diskon", "metode_bayar", "id_anggota"]
transaksi = pd.concat(                                    #<5>
    [kasir[kolom].assign(channel="gerai", sumber="kasir"),
     app[kolom + ["channel"]].assign(sumber="aplikasi")],
    ignore_index=True)

print(transaksi.shape)
print(transaksi.isna().sum().to_string())                 #<6>
print(transaksi["sumber"].value_counts().to_string())
transaksi.to_csv(DATA + "transaksi_gabungan_2026q1.csv",
                 index=False)
