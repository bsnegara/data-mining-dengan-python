import pandas as pd

DATA = "kode/data/"
trx = pd.read_csv(DATA + "transaksi_lengkap_2026q1.csv",
                  parse_dates=["waktu"])
trx["bulan"] = trx["waktu"].dt.strftime("%Y-%m")
trx["jam"] = trx["waktu"].dt.hour

kubus = trx.pivot_table(index="kota", columns="bulan",    #<1>
                        values="nilai", aggfunc="sum")
print((kubus / 1e6).round(1).to_string())                 #<2>

print((kubus.sum() / 1e6).round(1).to_string())           #<3>

pku = trx[trx["kota"] == "Pekanbaru"]                     #<4>
print((pku.pivot_table(index="kode_gerai", columns="bulan",
                       values="nilai", aggfunc="sum")
       / 1e6).round(1).to_string())

ramadan = trx["waktu"].between("2026-02-18", "2026-03-19 23:59")
irisan = trx[trx["bulan"] == "2026-01"]                   #<5>
print(irisan.groupby("kategori")["nilai"].sum()
      .div(1e6).round(1).to_string())
dadu = trx[ramadan & (trx["jam"] >= 17)                   #<6>
           & trx["kota"].isin(["Pekanbaru", "Padang"])]
print(dadu.pivot_table(index="kota", columns="kategori",
                       values="nilai", aggfunc="sum")
      .div(1e6).round(1).to_string())
