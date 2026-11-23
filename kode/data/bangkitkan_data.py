"""Pembangkit dataset sintetis Kopi Nusantara.

Seluruh perusahaan, gerai, menu, dan transaksi adalah REKAAN. Data
dibangkitkan dengan benih acak tetap sehingga hasilnya selalu sama.

Bagian yang sudah tersedia (dipakai Bab 1 dan 3):
  gerai.csv                     40 gerai di 8 kota
  menu.csv                      22 menu
  transaksi_kasir_2026q1.csv    baris item struk dari mesin kasir
  pesanan_aplikasi_2026q1.csv   baris item pesanan dari aplikasi
Kedua sumber transaksi sengaja memakai skema berbeda (nama kolom, kode
gerai, format waktu, nama metode bayar) seperti data dari dua sistem
yang dibangun oleh vendor berbeda.

Pemakaian:  python bangkitkan_data.py [folder_keluaran]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENIH = 2026
AWAL, AKHIR = "2026-01-01", "2026-03-31"
RAMADAN = ("2026-02-18", "2026-03-19")      # perkiraan 1 Ramadan s.d. akhir Ramadan
LEBARAN = ("2026-03-20", "2026-03-21")      # perkiraan Idulfitri 1447 H: gerai tutup

KOTA = [  # kota, provinsi, kode, jumlah gerai, faktor ramai, porsi aplikasi
    ("Pekanbaru", "Riau", "PKU", 8, 1.00, 0.22),
    ("Padang", "Sumatera Barat", "PDG", 5, 0.85, 0.18),
    ("Medan", "Sumatera Utara", "MDN", 6, 1.05, 0.25),
    ("Palembang", "Sumatera Selatan", "PLM", 4, 0.90, 0.22),
    ("Batam", "Kepulauan Riau", "BTM", 3, 0.95, 0.27),
    ("Jakarta", "DKI Jakarta", "JKT", 6, 1.35, 0.40),
    ("Bandung", "Jawa Barat", "BDG", 4, 1.10, 0.30),
    ("Yogyakarta", "DI Yogyakarta", "YOG", 4, 1.00, 0.24),
]
TIPE = {"kampus": 1.10, "perkantoran": 1.00, "mal": 1.20, "permukiman": 0.80}

MENU = [  # kode, nama, kategori, harga
    ("K01", "Kopi Susu Gula Aren", "kopi susu", 22000),
    ("K02", "Kopi Susu Pandan", "kopi susu", 24000),
    ("K03", "Kopi Susu Klepon", "kopi susu", 25000),
    ("K04", "Cafe Latte", "kopi susu", 28000),
    ("K05", "Cappuccino", "kopi susu", 28000),
    ("K06", "Es Kopi Susu Kelapa", "kopi susu", 26000),
    ("H01", "Americano", "kopi hitam", 20000),
    ("H02", "Espresso", "kopi hitam", 18000),
    ("H03", "V60 Gayo", "kopi hitam", 30000),
    ("H04", "V60 Kerinci", "kopi hitam", 30000),
    ("H05", "Kopi Tubruk Toraja", "kopi hitam", 18000),
    ("N01", "Teh Tarik", "non-kopi", 18000),
    ("N02", "Cokelat Aceh", "non-kopi", 24000),
    ("N03", "Matcha Latte", "non-kopi", 27000),
    ("N04", "Es Teh Serai", "non-kopi", 15000),
    ("N05", "Es Kelapa Muda", "non-kopi", 20000),
    ("M01", "Roti Bakar Srikaya", "makanan", 18000),
    ("M02", "Pisang Goreng Keju", "makanan", 20000),
    ("M03", "Croissant Mentega", "makanan", 22000),
    ("M04", "Nasi Goreng Kampung", "makanan", 32000),
    ("M05", "Kue Lapis Legit", "makanan", 15000),
    ("M06", "Bolu Pandan", "makanan", 12000),
]
# bobot popularitas dasar tiap menu (urutan sama dengan MENU)
BOBOT_MENU = np.array([18, 7, 5, 6, 5, 6, 6, 2, 2, 2, 3, 5, 4, 4, 5, 3, 4, 4, 3, 3, 2, 2], float)

# pola jam (07.00--22.00), hari biasa dan Ramadan
JAM = np.arange(7, 23)
POLA_JAM_BIASA = np.array([6, 9, 7, 5, 5, 8, 9, 6, 7, 9, 8, 6, 7, 8, 6, 3], float)
POLA_JAM_RAMADAN = np.array([1, 1, 1, 1, 1, 1, 2, 3, 5, 9, 14, 16, 12, 10, 8, 4], float)
FAKTOR_HARI = np.array([0.95, 0.92, 0.95, 0.97, 1.08, 1.30, 1.22])   # Senin..Minggu


def buat_gerai(rng):
    baris = []
    nomor = 1
    for kota, prov, kode, n, faktor, porsi_app in KOTA:
        tipe_kota = rng.choice(list(TIPE), size=n, p=[0.3, 0.25, 0.25, 0.2])
        if kota == "Pekanbaru":
            tipe_kota[0] = "kampus"            # gerai pertama: dekat kampus
        for i in range(n):
            buka = (pd.Timestamp("2016-08-17") if nomor == 1 else
                    pd.Timestamp("2017-01-01") + pd.Timedelta(int(rng.integers(0, 3100)), unit="D"))
            baris.append({
                "id_gerai": f"G{nomor:02d}",
                "kode_aplikasi": f"KN-{kode}-{i + 1:02d}",
                "nama_gerai": f"Kopi Nusantara {kota} {i + 1}",
                "kota": kota, "provinsi": prov, "tipe_lokasi": tipe_kota[i],
                "tanggal_buka": buka.date().isoformat(),
                "luas_m2": int(np.clip(rng.normal(90, 25), 40, 180)),
                "jumlah_kursi": int(np.clip(rng.normal(36, 10), 12, 70)),
            })
            nomor += 1
    g = pd.DataFrame(baris)
    g = g.sort_values("tanggal_buka").reset_index(drop=True)   # G01 = gerai tertua
    g["id_gerai"] = [f"G{i + 1:02d}" for i in range(len(g))]
    # nomor urut per kota mengikuti urutan pembukaan
    ke = g.groupby("kota").cumcount() + 1
    kode = g["kode_aplikasi"].str[3:6]
    g["kode_aplikasi"] = "KN-" + kode + "-" + ke.map("{:02d}".format)
    g["nama_gerai"] = "Kopi Nusantara " + g["kota"] + " " + ke.astype(str)
    return g


def bangkitkan_transaksi(rng, gerai, menu):
    tanggal = pd.date_range(AWAL, AKHIR, freq="D")
    ram0, ram1 = map(pd.Timestamp, RAMADAN)
    leb0, leb1 = map(pd.Timestamp, LEBARAN)
    info_kota = {k[0]: k for k in KOTA}
    harga = menu["harga"].to_numpy()
    kategori = menu["kategori"].to_numpy()
    kasir, aplikasi = [], []
    for _, gr in gerai.iterrows():
        _, _, _, _, f_kota, porsi_app = info_kota[gr["kota"]]
        f_tipe = TIPE[gr["tipe_lokasi"]]
        kekuatan = rng.lognormal(0, 0.15)                 # keunikan tiap gerai
        for t in tanggal:
            if leb0 <= t <= leb1:
                continue                                   # tutup saat Idulfitri
            ramadan = ram0 <= t <= ram1
            payday = t.day >= 25 or t.day <= 2
            lam = 34 * f_kota * f_tipe * kekuatan * FAKTOR_HARI[t.dayofweek]
            lam *= (0.85 if ramadan else 1.0) * (1.06 if payday else 1.0)
            if gr["tipe_lokasi"] == "kampus" and t < pd.Timestamp("2026-02-02"):
                lam *= 0.75                                # libur semester
            n = rng.poisson(lam)
            if n == 0:
                continue
            pola = POLA_JAM_RAMADAN if ramadan else POLA_JAM_BIASA
            jam = rng.choice(JAM, size=n, p=pola / pola.sum())
            detik = rng.integers(0, 3600, size=n)
            waktu = t + pd.to_timedelta(jam * 3600 + detik, unit="s")
            # porsi aplikasi naik pelan dari Januari ke Maret
            p_app = porsi_app * (1 + 0.15 * (t.month - 1) / 2)
            lewat_app = rng.random(n) < p_app
            n_item = 1 + rng.poisson(0.55, size=n)
            for j in range(n):
                w = BOBOT_MENU.copy()
                if jam[j] < 10:
                    w[kategori == "kopi hitam"] *= 1.8
                    w[menu["kode_menu"].isin(["M01", "M03"]).to_numpy()] *= 2.0
                if ramadan and jam[j] >= 17:
                    w[kategori == "non-kopi"] *= 1.8
                    w[kategori == "makanan"] *= 1.6
                idx = rng.choice(len(menu), size=n_item[j], replace=False, p=w / w.sum())
                qty = 1 + rng.poisson(0.15, size=len(idx))
                anggota = (f"A{rng.integers(1, 30001):05d}"
                           if rng.random() < (0.7 if lewat_app[j] else 0.35) else None)
                for k, q in zip(idx, qty):
                    rekaman = (waktu[j], gr, menu.iloc[k], q, harga[k], anggota)
                    (aplikasi if lewat_app[j] else kasir).append(rekaman + (j,))
    return kasir, aplikasi


def ke_dataframe(rng, kasir, aplikasi):
    # --- mesin kasir: skema berbahasa Indonesia, waktu lokal tanpa zona
    bayar = np.array(["Tunai", "QRIS", "Debit", "Kartu Kredit"])
    df_k = pd.DataFrame({
        "waktu": [r[0] for r in kasir],
        "kode_gerai": [r[1]["id_gerai"] for r in kasir],
        "kode_menu": [r[2]["kode_menu"] for r in kasir],
        "qty": [int(r[3]) for r in kasir],
        "harga_satuan": [int(r[4]) for r in kasir],
        "id_anggota": [r[5] for r in kasir],
        "_kunci": [f"{r[1]['id_gerai']}|{r[0]}|{r[6]}" for r in kasir],
    })
    kunci = df_k["_kunci"].unique()
    urut = sorted(kunci, key=lambda k: (k.split("|")[1], k))   # kronologis
    nomor = {k: f"S{i + 1:07d}" for i, k in enumerate(urut)}
    df_k["no_struk"] = df_k["_kunci"].map(nomor)
    metode = dict(zip(kunci, rng.choice(bayar, size=len(kunci), p=[0.30, 0.48, 0.15, 0.07])))
    df_k["metode_bayar"] = df_k["_kunci"].map(metode)
    df_k["diskon"] = 0
    promo = df_k["id_anggota"].notna() & (rng.random(len(df_k)) < 0.12)
    df_k.loc[promo, "diskon"] = (df_k.loc[promo, "harga_satuan"] * 0.2).astype(int)
    df_k = df_k.sort_values(["waktu", "no_struk"]).drop(columns="_kunci")
    df_k["waktu"] = df_k["waktu"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_k = df_k[["no_struk", "waktu", "kode_gerai", "kode_menu", "qty",
                 "harga_satuan", "diskon", "metode_bayar", "id_anggota"]]

    # --- aplikasi: skema berbahasa Inggris, waktu ISO 8601 dengan zona +07:00
    bayar_app = np.array(["e-wallet", "qris", "card", "virtual_account"])
    df_a = pd.DataFrame({
        "created_at": [r[0] for r in aplikasi],
        "outlet": [r[1]["kode_aplikasi"] for r in aplikasi],
        "item_name": [r[2]["nama_menu"] for r in aplikasi],
        "quantity": [int(r[3]) for r in aplikasi],
        "unit_price": [int(r[4]) for r in aplikasi],
        "member_id": [r[5] for r in aplikasi],
        "_kunci": [f"{r[1]['id_gerai']}|{r[0]}|{r[6]}" for r in aplikasi],
    })
    kunci = df_a["_kunci"].unique()
    urut = sorted(kunci, key=lambda k: (k.split("|")[1], k))
    nomor = {k: f"ORD-{i + 1:06d}" for i, k in enumerate(urut)}
    df_a["order_id"] = df_a["_kunci"].map(nomor)
    metode = dict(zip(kunci, rng.choice(bayar_app, size=len(kunci), p=[0.55, 0.20, 0.10, 0.15])))
    kanal = dict(zip(kunci, rng.choice(["delivery", "pickup"], size=len(kunci), p=[0.7, 0.3])))
    df_a["payment"] = df_a["_kunci"].map(metode)
    df_a["channel"] = df_a["_kunci"].map(kanal)
    df_a["promo_amount"] = 0
    promo = rng.random(len(df_a)) < 0.25
    df_a.loc[promo, "promo_amount"] = (df_a.loc[promo, "unit_price"] * 0.25).astype(int)
    df_a = df_a.sort_values(["created_at", "order_id"]).drop(columns="_kunci")
    df_a["created_at"] = df_a["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%S+07:00")
    df_a = df_a[["order_id", "created_at", "outlet", "item_name", "quantity",
                 "unit_price", "promo_amount", "payment", "channel", "member_id"]]
    return df_k, df_a


def main(folder="."):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BENIH)
    gerai = buat_gerai(rng)
    menu = pd.DataFrame(MENU, columns=["kode_menu", "nama_menu", "kategori", "harga"])
    kasir, aplikasi = bangkitkan_transaksi(rng, gerai, menu)
    df_k, df_a = ke_dataframe(rng, kasir, aplikasi)
    gerai.to_csv(folder / "gerai.csv", index=False)
    menu.to_csv(folder / "menu.csv", index=False)
    df_k.to_csv(folder / "transaksi_kasir_2026q1.csv", index=False)
    df_a.to_csv(folder / "pesanan_aplikasi_2026q1.csv", index=False)
    print(f"gerai {len(gerai)}, menu {len(menu)}, kasir {len(df_k)} baris, "
          f"aplikasi {len(df_a)} baris")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
