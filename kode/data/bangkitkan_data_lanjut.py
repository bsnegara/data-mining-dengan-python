"""Pembangkit dataset sintetis Kopi Nusantara, bagian kedua.

Melengkapi bangkitkan_data.py (gerai, menu, transaksi Q1 2026). Seluruh
data adalah REKAAN dengan benih acak tetap. Hubungan antaratribut sengaja
dirancang agar setiap bab menemukan pola yang bermakna, tetapi tetap
mengandung derau, nilai kosong, dan kasus sulit seperti data sungguhan.

Keluaran (folder yang sama):
  pelanggan.csv            8.000 anggota loyalitas, atribut RFM, churn
  pelanggan_mentah.csv     versi kotor dari pelanggan.csv (Bab 5)
  mutu_biji.csv            1.500 kiriman biji dari petani mitra
  profil_rasa.csv          24 kopi asal Nusantara x 12 atribut rasa
  penjualan_harian.csv     omzet harian per gerai 2023-01-01..2026-03-31
  batch_sangrai.csv        ringkasan 80 batch sangrai
  sensor_sangrai.csv       suhu biji dan lingkungan per detik
  audit_kasir.csv          ringkasan harian per kasir, Q1 2026
  sesi_aplikasi.csv        urutan kejadian dalam 5.000 sesi aplikasi
  ulasan.csv               3.000 ulasan pelanggan berbahasa Indonesia

Pemakaian:  python bangkitkan_data_lanjut.py [folder]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENIH = 2027
POTONG = pd.Timestamp("2026-03-31")         # tanggal potong data anggota

KOTA_BOBOT = {"Pekanbaru": .20, "Padang": .11, "Medan": .15, "Palembang": .09,
              "Batam": .07, "Jakarta": .19, "Bandung": .10, "Yogyakarta": .09}

# Ramadan dan Idulfitri (perkiraan kalender pemerintah)
RAMADAN = [("2023-03-23", "2023-04-20"), ("2024-03-11", "2024-04-09"),
           ("2025-03-01", "2025-03-30"), ("2026-02-18", "2026-03-19")]
LEBARAN = [("2023-04-22", "2023-04-23"), ("2024-04-10", "2024-04-11"),
           ("2025-03-31", "2025-04-01"), ("2026-03-20", "2026-03-21")]
LIBUR = ["2023-01-01", "2023-01-22", "2023-02-18", "2023-03-22", "2023-04-07",
         "2023-05-01", "2023-05-18", "2023-06-01", "2023-06-04", "2023-06-29",
         "2023-07-19", "2023-08-17", "2023-09-28", "2023-12-25",
         "2024-01-01", "2024-02-08", "2024-02-10", "2024-02-14", "2024-03-11",
         "2024-03-29", "2024-05-01", "2024-05-09", "2024-05-23", "2024-06-01",
         "2024-06-17", "2024-07-07", "2024-08-17", "2024-09-16", "2024-12-25",
         "2025-01-01", "2025-01-27", "2025-01-29", "2025-03-29", "2025-04-18",
         "2025-05-01", "2025-05-12", "2025-05-29", "2025-06-01", "2025-06-06",
         "2025-06-27", "2025-08-17", "2025-09-05", "2025-12-25",
         "2026-01-01", "2026-01-16", "2026-02-17", "2026-03-19"]


def logistik(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------
# 1. Anggota program loyalitas
# ---------------------------------------------------------------------
SEGMEN = {  # nama: (porsi, umur rata, umur sb, kunjungan/90h, nilai struk, pagi, akhir pekan, kopi hitam, aplikasi, promo)
    "mahasiswa": (.34, 21, 2.2, 14, 31000, .35, .30, .10, .30, .45),
    "pekerja":   (.33, 31, 5.0, 20, 38000, .70, .12, .25, .55, .20),
    "keluarga":  (.18, 39, 7.0, 6, 92000, .15, .70, .08, .35, .25),
    "penikmat":  (.15, 34, 8.0, 17, 46000, .45, .35, .70, .25, .10),
}


def buat_pelanggan(rng, gerai, n=8000):
    nama_seg = list(SEGMEN)
    seg = rng.choice(nama_seg, n, p=[SEGMEN[s][0] for s in nama_seg])
    kota = rng.choice(list(KOTA_BOBOT), n, p=list(KOTA_BOBOT.values()))
    baris = []
    for i in range(n):
        s = SEGMEN[seg[i]]
        g = gerai[gerai.kota == kota[i]]
        if seg[i] == "mahasiswa" and (g.tipe_lokasi == "kampus").any():
            g = g[g.tipe_lokasi == "kampus"] if rng.random() < .7 else g
        if seg[i] == "pekerja" and (g.tipe_lokasi == "perkantoran").any():
            g = g[g.tipe_lokasi == "perkantoran"] if rng.random() < .6 else g
        gu = g.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        buka = max(pd.Timestamp(gu.tanggal_buka), pd.Timestamp("2018-01-01"))
        hari = (POTONG - pd.Timedelta(120, unit="D") - buka).days
        daftar = buka + pd.Timedelta(int(hari * rng.beta(1.3, 1.0)), unit="D")
        tenure = (POTONG - daftar).days / 30.44
        umur = int(np.clip(rng.normal(s[1], s[2]), 17, 70))
        jk = rng.choice(["L", "P"], p=[.46, .54])
        keterlibatan = rng.gamma(4, .25)                 # rata-rata 1
        promo = float(np.clip(rng.beta(2, 2) * 2 * s[9], 0, 1))
        keluhan = rng.poisson(.25 + .4 * (keterlibatan > 1.4))
        jarak = float(np.round(rng.gamma(2, 1.3) + (2 if seg[i] == "keluarga" else 0), 1))
        # risiko churn laten
        z = (-2.75 - .9 * np.log(keterlibatan) + 1.9 * (promo - .25) + .55 * keluhan
             - .35 * np.log1p(tenure / 12) + .12 * jarak
             + {"mahasiswa": .35, "pekerja": -.1, "keluarga": .15, "penikmat": -.55}[seg[i]]
             + rng.normal(0, .75))
        p_churn = logistik(z)
        churn = rng.random() < p_churn
        # perilaku 90 hari terakhir; anggota yang akan churn sudah melemah
        melemah = rng.uniform(.25, .7) if churn else rng.uniform(.85, 1.15)
        frek = int(rng.poisson(s[3] * keterlibatan * melemah))
        frek_lalu = int(rng.poisson(s[3] * keterlibatan))
        if frek == 0:
            recency = int(rng.integers(91, 150))
        else:
            skala = 90 / (frek + 1) * (2.2 if churn else 1.0)
            recency = int(min(90, rng.exponential(skala)))
        nilai_struk = s[4] * rng.lognormal(0, .22)
        nilai = int(round(frek * nilai_struk, -2))
        belanja_depan = 0 if churn else int(round(
            max(0, (nilai / 3) * rng.lognormal(0, .25) * (1.08 if tenure > 24 else .97)
                + rng.normal(0, 15000)), -2))
        baris.append({
            "id_anggota": f"A{i + 1:05d}", "kota": kota[i], "gerai_utama": gu.id_gerai,
            "tanggal_daftar": daftar.date().isoformat(), "umur": umur, "jenis_kelamin": jk,
            "frekuensi_90h": frek, "frekuensi_90h_lalu": frek_lalu, "recency_hari": recency,
            "nilai_90h": nilai, "rata_nilai_struk": int(round(nilai_struk, -2)) if frek else 0,
            "porsi_promo": round(promo, 2),
            "porsi_aplikasi": round(float(np.clip(rng.beta(2, 2) * 2 * s[8], 0, 1)), 2),
            "porsi_kopi_hitam": round(float(np.clip(rng.beta(2, 3) * 2.2 * s[7], 0, 1)), 2),
            "porsi_pagi": round(float(np.clip(rng.normal(s[5], .12), 0, 1)), 2),
            "porsi_akhir_pekan": round(float(np.clip(rng.normal(s[6], .12), 0, 1)), 2),
            "ragam_menu": int(np.clip(rng.poisson(2 + .25 * frek), 1, 22)) if frek else 0,
            "keluhan_90h": int(keluhan), "jarak_km": jarak,
            "pakai_ulasan": int(rng.random() < logistik(-1.6 + .8 * keterlibatan - 1.2 * churn)),
            "segmen_laten": seg[i], "churn": int(churn), "belanja_bulan_depan": belanja_depan,
        })
    df = pd.DataFrame(baris)
    total = df.nilai_90h
    df["tingkat"] = np.where(total >= total.quantile(.85), "emas",
                             np.where(total >= total.quantile(.5), "perak", "perunggu"))
    urut = ["id_anggota", "kota", "gerai_utama", "tanggal_daftar", "umur", "jenis_kelamin",
            "tingkat", "frekuensi_90h", "frekuensi_90h_lalu", "recency_hari", "nilai_90h",
            "rata_nilai_struk", "porsi_promo", "porsi_aplikasi", "porsi_kopi_hitam", "porsi_pagi",
            "porsi_akhir_pekan", "ragam_menu", "keluhan_90h", "jarak_km", "pakai_ulasan",
            "segmen_laten", "churn", "belanja_bulan_depan"]
    return df[urut]


def kotori_pelanggan(rng, df):
    """Versi mentah: nilai kosong, salah ketik, format campur, duplikat."""
    m = df.drop(columns=["segmen_laten", "belanja_bulan_depan"]).copy()
    n = len(m)
    m["umur"] = m["umur"].astype("object")
    idx = rng.choice(n, int(.07 * n), replace=False)
    m.loc[idx, "umur"] = np.nan
    m.loc[rng.choice(n, 9, replace=False), "umur"] = [0, 2, 99, 150, 1, 0, 120, 3, 200]
    m.loc[rng.choice(n, int(.03 * n), replace=False), "jenis_kelamin"] = np.nan
    varian = {"Pekanbaru": ["pekanbaru", "Pekan Baru", "PEKANBARU"],
              "Yogyakarta": ["Jogja", "Yogya", "DIY"], "Jakarta": ["DKI Jakarta", "jakarta"],
              "Medan": ["medan "], "Bandung": ["Bdg"]}
    for k, vs in varian.items():
        cocok = np.where(m.kota == k)[0]
        pilih = rng.choice(cocok, int(.06 * len(cocok)), replace=False)
        m.loc[pilih, "kota"] = rng.choice(vs, len(pilih))
    # format tanggal campuran
    pilih = rng.choice(n, int(.05 * n), replace=False)
    m.loc[pilih, "tanggal_daftar"] = pd.to_datetime(m.loc[pilih, "tanggal_daftar"]).dt.strftime("%d/%m/%Y")
    # nilai belanja sebagai teks rupiah pada sebagian baris
    m["nilai_90h"] = m["nilai_90h"].astype("object")
    pilih = rng.choice(n, int(.04 * n), replace=False)
    m.loc[pilih, "nilai_90h"] = ["Rp" + f"{int(v):,}".replace(",", ".") for v in m.loc[pilih, "nilai_90h"]]
    # jarak salah satuan (meter)
    pilih = rng.choice(n, 25, replace=False)
    m.loc[pilih, "jarak_km"] = (m.loc[pilih, "jarak_km"] * 1000).round()
    m.loc[rng.choice(n, int(.02 * n), replace=False), "porsi_aplikasi"] = np.nan
    dup = m.sample(64, random_state=7)
    m = pd.concat([m, dup]).sample(frac=1, random_state=8).reset_index(drop=True)
    return m


# ---------------------------------------------------------------------
# 2. Mutu biji kiriman petani
# ---------------------------------------------------------------------
ASAL = {  # asal: (porsi, mdpl, sb, proses, profil rasa [aroma, flavor, aftertaste, acidity, body, balance, sweetness])
    "Gayo":    (.32, 1450, 120, {"giling basah": .65, "full washed": .2, "natural": .1, "honey": .05},
                [7.6, 7.6, 7.4, 7.2, 8.1, 7.5, 7.5]),
    "Kerinci": (.26, 1500, 140, {"giling basah": .35, "full washed": .3, "natural": .2, "honey": .15},
                [7.7, 7.6, 7.3, 7.9, 7.3, 7.4, 7.6]),
    "Toraja":  (.22, 1550, 150, {"giling basah": .45, "full washed": .4, "natural": .1, "honey": .05},
                [7.5, 7.6, 7.5, 7.5, 7.7, 7.8, 7.5]),
    "Flores":  (.20, 1400, 130, {"giling basah": .2, "full washed": .55, "natural": .15, "honey": .1},
                [7.6, 7.7, 7.5, 7.4, 7.7, 7.6, 8.0]),
}
KELOMPOK = {"Gayo": ["KT Bener Meriah", "KT Takengon", "KT Atu Lintang"],
            "Kerinci": ["KT Kayu Aro", "KT Gunung Tujuh", "KT Siulak"],
            "Toraja": ["KT Sapan", "KT Pulu-Pulu", "KT Awan"],
            "Flores": ["KT Bajawa", "KT Ruteng", "KT Colol"]}
VARIETAS = ["Ateng", "Sigarar Utang", "Typica", "S795", "Abesinia"]


def buat_mutu_biji(rng, n=1500):
    asal = rng.choice(list(ASAL), n, p=[ASAL[a][0] for a in ASAL])
    tgl = pd.Timestamp("2023-01-02") + pd.to_timedelta(rng.integers(0, 1090, n), unit="D")
    baris = []
    for i in range(n):
        a = ASAL[asal[i]]
        bulan = tgl[i].month
        musim = "hujan" if bulan in (11, 12, 1, 2, 3, 4) else "kemarau"
        proses = rng.choice(list(a[3]), p=list(a[3].values()))
        jemur = rng.choice(["para-para", "terpal", "mesin"], p=[.45, .4, .15])
        mdpl = int(rng.normal(a[1], a[2]))
        simpan = int(rng.gamma(2.2, 22))
        kadar_air = (11.4 + (.9 if musim == "hujan" else 0) + (.7 if proses == "giling basah" else 0)
                     + {"para-para": 0, "terpal": .5, "mesin": -.4}[jemur] + rng.normal(0, .55))
        kadar_air = round(float(np.clip(kadar_air, 9.5, 15.5)), 1)
        p_jamur = logistik(-2.7 + 1.6 * (kadar_air - 12.5) + .018 * (simpan - 45))
        jamur = rng.random() < p_jamur
        cacat_p = int(rng.poisson(1.2 + 6 * jamur + .5 * (jemur == "terpal")))
        cacat_s = int(rng.poisson(6 + 3 * jamur + .04 * simpan))
        screen = int(np.clip(round(rng.normal(16.2 + (mdpl - 1400) / 400, .9)), 13, 19))
        dens = int(rng.normal(640 + .06 * (mdpl - 1400) - 6 * (kadar_air - 12), 18))
        mutu = (.0035 * (mdpl - 1400) - .09 * cacat_p - .025 * cacat_s - .35 * jamur
                + {"giling basah": 0, "full washed": .12, "natural": .18, "honey": .15}[proses]
                + rng.normal(0, .22))
        prof = np.array(a[4]) - .3 + mutu + rng.normal(0, .18, 7)
        unif = 10 - 2 * rng.binomial(2, .06 + .2 * jamur)
        clean = 10 - 2 * rng.binomial(2, .05 + .25 * jamur)
        overall = np.mean(prof) + rng.normal(0, .15)
        atr = np.clip(np.round(np.r_[prof[:6], overall] * 4) / 4, 6, 9.5)
        sweet = 10 - 2 * rng.binomial(1, .04 + .2 * jamur)
        total = round(float(atr.sum() + unif + clean + sweet - 2 * (cacat_p > 5)), 2)
        # keputusan juru uji: aturan baku plus pertimbangan (derau)
        skor_terima = (-(kadar_air - 12.5) * 2.2 - (cacat_p - 5) * .55 + (total - 79) * .55
                       + rng.normal(0, .9))
        baris.append({
            "id_kiriman": f"K-{i + 1:04d}", "tanggal": tgl[i].date().isoformat(),
            "asal": asal[i], "kelompok_tani": rng.choice(KELOMPOK[asal[i]]),
            "varietas": rng.choice(VARIETAS, p=[.35, .25, .15, .15, .10]),
            "proses": proses, "musim_panen": musim, "cara_jemur": jemur,
            "ketinggian_mdpl": mdpl, "lama_simpan_hari": simpan, "kadar_air": kadar_air,
            "cacat_primer": cacat_p, "cacat_sekunder": cacat_s, "ukuran_screen": screen,
            "densitas_gl": dens, "jamur": int(jamur),
            "aroma": atr[0], "flavor": atr[1], "aftertaste": atr[2], "acidity": atr[3],
            "body": atr[4], "balance": atr[5], "uniformity": unif, "clean_cup": clean,
            "sweetness": sweet, "overall": atr[6], "skor_total": total,
            "specialty": int(total >= 80), "keputusan": "terima" if skor_terima > 0 else "tolak",
        })
    return pd.DataFrame(baris)


PROFIL = {  # 24 kopi asal (panel rasa rekaan) -> koordinat laten (intensitas, kecerahan, manis)
    "Gayo": (1.2, -.2, .1), "Mandailing": (1.4, -.6, 0), "Lintong": (1.1, -.4, -.2),
    "Sidikalang": (1.3, -.8, -.3), "Kerinci": (.2, 1.0, .3), "Solok": (.1, .8, .2),
    "Semendo": (.7, -.3, -.4), "Lampung Robusta": (1.6, -1.3, -.9), "Preanger": (-.2, .9, .2),
    "Malabar": (-.1, .7, .4), "Temanggung": (.6, -.5, -.6), "Ijen": (-.3, .6, .5),
    "Kintamani": (-.8, 1.3, .6), "Toraja": (.5, .3, .1), "Kalosi": (.6, .1, -.1),
    "Enrekang": (.4, .4, 0), "Bajawa": (.3, .1, .9), "Manggarai": (.2, .2, .7),
    "Wamena": (-.4, .5, .8), "Pegunungan Bintang": (-.5, .6, .6), "Halmahera": (.0, -.2, .3),
    "Malino": (.1, .2, .2), "Aceh Robusta": (1.5, -1.1, -.7), "Dampit": (1.3, -1.2, -.8),
}
ATR_RASA = ["aroma", "cokelat", "rempah", "tanah", "buah", "bunga", "sitrus", "keasaman",
            "manis", "kekentalan", "pahit", "rasa_ikutan"]
MUAT = np.array([  # muatan setiap atribut pada (intensitas, kecerahan, manis)
    [.2, .4, .4], [.6, -.2, .5], [.7, -.1, -.1], [.9, -.6, -.3], [-.3, .8, .4],
    [-.5, .7, .2], [-.4, .9, -.1], [-.3, .9, -.2], [-.1, .2, .9], [.9, -.3, .2],
    [.8, -.4, -.6], [.5, .2, .5]])


def buat_profil_rasa(rng):
    nama = list(PROFIL)
    Z = np.array([PROFIL[k] for k in nama])
    X = 6 + Z @ MUAT.T * 1.1 + rng.normal(0, .25, (len(nama), len(ATR_RASA)))
    df = pd.DataFrame(np.round(np.clip(X, 2, 10), 1), columns=ATR_RASA)
    df.insert(0, "kopi", nama)
    pulau = {"Sumatera": ["Gayo", "Mandailing", "Lintong", "Sidikalang", "Kerinci", "Solok",
                          "Semendo", "Lampung Robusta", "Aceh Robusta"],
             "Jawa": ["Preanger", "Malabar", "Temanggung", "Ijen", "Dampit"],
             "Bali-Nusa Tenggara": ["Kintamani", "Bajawa", "Manggarai"],
             "Sulawesi": ["Toraja", "Kalosi", "Enrekang", "Malino"],
             "Papua-Maluku": ["Wamena", "Pegunungan Bintang", "Halmahera"]}
    peta = {k: p for p, ks in pulau.items() for k in ks}
    df.insert(1, "pulau", df.kopi.map(peta))
    df.insert(2, "spesies", np.where(df.kopi.str.contains("Robusta|Dampit"), "robusta", "arabika"))
    return df


# ---------------------------------------------------------------------
# 3. Penjualan harian per gerai
# ---------------------------------------------------------------------
def dalam(tgl, rentang):
    hasil = np.zeros(len(tgl), bool)
    for a, b in rentang:
        hasil |= (tgl >= a) & (tgl <= b)
    return hasil


def buat_penjualan_harian(rng, gerai, folder):
    tgl = pd.date_range("2023-01-01", "2026-03-31", freq="D")
    ram = dalam(tgl, RAMADAN)
    leb = dalam(tgl, LEBARAN)
    pasca = dalam(tgl, [(pd.Timestamp(b) + pd.Timedelta(1, unit="D"),
                         pd.Timestamp(b) + pd.Timedelta(7, unit="D")) for _, b in LEBARAN])
    libur = tgl.isin(pd.to_datetime(LIBUR))
    dow = np.array([.92, .95, .97, 1.0, 1.08, 1.18, 1.10])[tgl.dayofweek]
    gajian = np.where(tgl.day >= 25, 1.06, np.where(tgl.day <= 3, 1.04, 1.0))
    musim_hujan = np.isin(tgl.month, [11, 12, 1, 2, 3, 4])
    hasil = []
    hujan_kota = {}
    for k in KOTA_BOBOT:
        p = np.where(musim_hujan, .55, .25)
        hujan_kota[k] = np.round(np.where(rng.random(len(tgl)) < p,
                                          rng.gamma(1.2, 14, len(tgl)), 0), 1)
    for _, g in gerai.iterrows():
        buka = pd.Timestamp(g.tanggal_buka)
        umur = (tgl - buka).days / 365.25
        ramp = np.clip(.55 + .45 * np.minimum(umur, 1), 0, 1)
        dasar = (900_000 + 6_500 * g.jumlah_kursi + 3_000 * g.luas_m2) * {
            "kampus": 1.0, "perkantoran": 1.05, "mal": 1.25, "permukiman": .8}[g.tipe_lokasi]
        dasar *= {"Jakarta": 1.35, "Bandung": 1.1, "Medan": 1.05}.get(g.kota, 1.0)
        tren = 1 + .07 * (tgl.year - 2023 + tgl.dayofyear / 365)
        efek_ram = np.where(ram, .86 if g.tipe_lokasi != "mal" else 1.04, 1.0)
        efek_libur = np.where(libur, 1.12 if g.tipe_lokasi in ("mal", "permukiman") else .78, 1.0)
        kampus_libur = (g.tipe_lokasi == "kampus") & (np.isin(tgl.month, [7, 8]) |
                                                      ((tgl.month == 1) & (tgl.day > 10)))
        hujan = hujan_kota[g.kota]
        efek_hujan = 1 - .004 * np.minimum(hujan, 60)
        mu = (dasar * ramp * tren * dow * gajian * efek_ram * efek_libur * efek_hujan
              * np.where(kampus_libur, .8, 1) * np.where(pasca, .82, 1))
        omzet = np.asarray(mu, float) * rng.lognormal(0, .09, len(tgl))
        omzet[leb] = 0
        omzet[np.asarray(tgl < buka)] = np.nan
        struk = np.round(omzet / rng.normal(38000, 1500, len(tgl)))
        df = pd.DataFrame({"tanggal": tgl, "id_gerai": g.id_gerai, "kota": g.kota,
                           "omzet": np.round(omzet, -2), "jumlah_struk": struk,
                           "hujan_mm": hujan, "libur_nasional": libur.astype(int),
                           "ramadan": ram.astype(int), "tutup_lebaran": leb.astype(int)})
        hasil.append(df.dropna(subset=["omzet"]))
    harian = pd.concat(hasil, ignore_index=True)
    # Q1 2026 diselaraskan dengan data transaksi Bab 3
    lengkap = folder / "transaksi_lengkap_2026q1.csv"
    if lengkap.exists():
        t = pd.read_csv(lengkap, usecols=["waktu", "kode_gerai", "no_struk", "nilai"])
        t["tanggal"] = pd.to_datetime(t.waktu.str[:10])
        agr = t.groupby(["tanggal", "kode_gerai"]).agg(omzet_t=("nilai", "sum"),
                                                       struk_t=("no_struk", "nunique")).reset_index()
        harian = harian.merge(agr, left_on=["tanggal", "id_gerai"],
                              right_on=["tanggal", "kode_gerai"], how="left")
        q1 = harian.tanggal >= "2026-01-01"
        # skala historis disamakan dengan skala transaksi Q1 2026
        rasio = (harian.loc[q1, "omzet_t"].sum() / harian.loc[q1, "omzet"].sum())
        harian["omzet"] = np.round(harian.omzet * rasio, -2)
        harian["jumlah_struk"] = np.round(harian.jumlah_struk * rasio)
        harian.loc[q1, "omzet"] = harian.loc[q1, "omzet_t"].fillna(0)
        harian.loc[q1, "jumlah_struk"] = harian.loc[q1, "struk_t"].fillna(0)
        harian = harian.drop(columns=["kode_gerai", "omzet_t", "struk_t"])
    harian["jumlah_struk"] = harian.jumlah_struk.astype(int)
    harian["omzet"] = harian.omzet.astype(int)
    # kebutuhan biji (kg): sekitar 15 gram per struk, sebagian menu nonkopi
    harian["biji_kg"] = np.round(harian.jumlah_struk * 1.6 * .015 * rng.normal(1, .04, len(harian)), 2)
    harian["tanggal"] = harian.tanggal.dt.date.astype(str)
    return harian


# ---------------------------------------------------------------------
# 4. Mesin sangrai
# ---------------------------------------------------------------------
def kurva_sangrai(rng, durasi, suhu_akhir, laju=1.0):
    t = np.arange(durasi + 1)
    tp = 85 + rng.normal(0, 5)                       # titik balik (detik)
    bt0 = 200 + rng.normal(0, 4)
    turun = 92 + rng.normal(0, 3)
    naik = turun + (suhu_akhir - turun) * (1 - np.exp(-laju * (t - tp) / (durasi * .55))) / (
        1 - np.exp(-laju * (durasi - tp) / (durasi * .55)))
    bt = np.where(t < tp, turun + (bt0 - turun) * np.exp(-t / 28), naik)
    et = 228 + 12 * np.tanh((t - 300) / 200) + rng.normal(0, 1.2, len(t))
    return t, bt, et


def buat_sangrai(rng, n=80):
    batch, sensor = [], []
    aneh = {17: "lonjakan sensor", 38: "macet (stall)", 55: "terlalu cepat", 71: "lonjakan sensor"}
    tgl0 = pd.Timestamp("2026-02-02")
    for b in range(1, n + 1):
        durasi = int(rng.normal(690, 25))
        akhir = rng.normal(210, 2.5)
        laju = 1.0
        jenis = aneh.get(b, "normal")
        if jenis == "terlalu cepat":
            durasi, laju = int(durasi * .72), 1.4
        t, bt, et = kurva_sangrai(rng, durasi, akhir, laju)
        if jenis == "macet (stall)":
            m = (t > 380) & (t < 470)
            bt[m] = bt[t == 380][0] + .02 * (t[m] - 380)
            bt[t >= 470] = bt[t >= 470] - (bt[t == 470][0] - bt[t == 469][0]) + .2
        bt = bt + rng.normal(0, .7, len(t))
        if jenis == "lonjakan sensor":
            for _ in range(6):
                j = int(rng.integers(150, len(t) - 10))
                bt[j:j + 2] += rng.choice([-1, 1]) * rng.uniform(18, 35)
        lewat = (t > 150) & (bt > 196)            # first crack setelah titik balik
        crack = int(t[np.argmax(lewat)]) if lewat.any() else durasi
        asal = rng.choice(list(ASAL))
        batch.append({"id_batch": f"B{b:03d}", "tanggal": (tgl0 + pd.Timedelta(b // 3, unit="D")).date().isoformat(),
                      "asal": asal, "berat_kg": round(float(rng.normal(12, .4)), 1),
                      "durasi_detik": durasi, "suhu_akhir": round(float(bt[-5:].mean()), 1),
                      "detik_crack": crack,
                      "hasil_uji": "cacat" if jenis in ("macet (stall)", "terlalu cepat") else "baik",
                      "catatan_operator": jenis if jenis != "lonjakan sensor" else "normal"})
        sensor.append(pd.DataFrame({"id_batch": f"B{b:03d}", "detik": t,
                                    "suhu_biji": np.round(bt, 1), "suhu_lingkungan": np.round(et, 1)}))
    return pd.DataFrame(batch), pd.concat(sensor, ignore_index=True)


# ---------------------------------------------------------------------
# 5. Audit kasir
# ---------------------------------------------------------------------
def buat_audit_kasir(rng, gerai):
    tgl = pd.date_range("2026-01-01", "2026-03-31", freq="D")
    tgl = tgl[~dalam(tgl, [LEBARAN[-1]])]
    curang = {"G07-K2": "batal", "G18-K1": "diskon", "G31-K3": "batal"}
    baris = []
    for _, g in gerai.iterrows():
        for k in range(1, 4):
            kid = f"{g.id_gerai}-K{k}"
            sifat = rng.normal(0, 1, 3)
            for d in tgl:
                if rng.random() < .28:           # kasir libur
                    continue
                struk = int(rng.poisson(85 * (1.15 if d.dayofweek >= 4 else 1)))
                p_batal = .012 * np.exp(.2 * sifat[0])
                diskon = rng.gamma(2, 9000) * np.exp(.2 * sifat[1])
                tunai = np.clip(rng.normal(.28 + .03 * sifat[2], .05), 0, 1)
                tutup = int(rng.poisson(.15))
                if curang.get(kid) == "batal" and d >= pd.Timestamp("2026-02-01"):
                    p_batal *= 4.5
                    tunai = min(1, tunai + .2)
                    tutup += int(rng.poisson(1.5))
                if curang.get(kid) == "diskon" and d >= pd.Timestamp("2026-01-20"):
                    diskon *= rng.uniform(3, 6)
                batal = int(rng.binomial(struk, p_batal))
                baris.append({"tanggal": d.date().isoformat(), "id_gerai": g.id_gerai, "id_kasir": kid,
                              "jumlah_struk": struk, "struk_batal": batal,
                              "diskon_manual": int(round(diskon, -2)),
                              "rata_nilai_struk": int(round(rng.normal(38500, 2500), -2)),
                              "porsi_tunai": round(float(tunai), 3), "struk_setelah_tutup": tutup})
    return pd.DataFrame(baris)


# ---------------------------------------------------------------------
# 6. Sesi aplikasi (rantai Markov dengan dua keadaan tersembunyi)
# ---------------------------------------------------------------------
KEJADIAN = ["beranda", "promo", "cari", "lihat_menu", "detail_menu", "keranjang", "bayar",
            "selesai", "keluar"]


def buat_sesi(rng, n=5000):
    # matriks transisi per keadaan tersembunyi: 0 = menjelajah, 1 = berniat beli
    T = {
        0: {"beranda": [0, .30, .15, .40, 0, .02, 0, 0, .13], "promo": [.25, 0, .10, .35, .10, .03, 0, 0, .17],
            "cari": [.10, .05, .10, .30, .30, .03, 0, 0, .12], "lihat_menu": [.12, .10, .08, .10, .35, .05, 0, 0, .20],
            "detail_menu": [.08, .05, .05, .40, .05, .15, 0, 0, .22], "keranjang": [.05, .05, .05, .25, .10, 0, .20, 0, .30],
            "bayar": [0, 0, 0, 0, 0, .25, 0, .45, .30]},
        1: {"beranda": [0, .08, .30, .55, 0, .05, 0, 0, .02], "promo": [.05, 0, .10, .45, .25, .13, 0, 0, .02],
            "cari": [0, .02, .05, .15, .70, .06, 0, 0, .02], "lihat_menu": [.02, .03, .05, .05, .55, .27, 0, 0, .03],
            "detail_menu": [.01, .01, .02, .20, .03, .70, 0, 0, .03], "keranjang": [0, .02, .02, .10, .03, 0, .80, 0, .03],
            "bayar": [0, 0, 0, 0, 0, .06, 0, .90, .04]},
    }
    pindah = {0: .12, 1: .05}           # peluang berganti keadaan per langkah
    anggota = [f"A{i:05d}" for i in rng.integers(1, 8001, n)]
    baris = []
    for s in range(n):
        h = int(rng.random() < .42)
        x = "beranda"
        waktu = pd.Timestamp("2026-03-01 06:00") + pd.Timedelta(int(rng.integers(0, 30 * 24 * 3600 - 7200)), unit="s")
        langkah = 0
        while True:
            baris.append({"id_sesi": f"S{s + 1:05d}", "id_anggota": anggota[s], "urutan": langkah + 1,
                          "kejadian": x, "waktu": waktu.isoformat(), "niat_laten": h})
            if x in ("selesai", "keluar") or langkah > 40:
                break
            x = rng.choice(KEJADIAN, p=T[h][x])
            if rng.random() < pindah[h]:
                h = 1 - h
            waktu += pd.Timedelta(int(rng.gamma(2, 9)), unit="s")
            langkah += 1
    return pd.DataFrame(baris)


# ---------------------------------------------------------------------
# 7. Ulasan pelanggan
# ---------------------------------------------------------------------
FRASA = {
    ("rasa", 1): ["kopinya enak banget", "kopi susu gula arennya mantap", "rasanya pas, tidak terlalu manis",
                  "v60 gayonya juara", "espressonya kental dan wangi", "rasanya konsisten dari dulu",
                  "kopinya mantul", "pandannya wangi bgt", "rasanya bikin nagih"],
    ("rasa", 0): ["kopinya hambar", "kemanisan, gulanya kebanyakan", "rasanya beda dari biasanya",
                  "kopinya encer kayak air", "rasanya pahit gosong", "es kopinya cuma rasa es",
                  "kopinya asem banget, kayak basi"],
    ("harga", 1): ["harganya terjangkau", "harga ramah kantong mahasiswa", "worth it dengan harganya",
                   "promonya sering ada", "murah tapi berkualitas"],
    ("harga", 0): ["kemahalan untuk ukuran segitu", "harganya naik terus", "mahal bgt sekarang",
                   "promonya cuma tipu-tipu", "ga sebanding sama harganya"],
    ("pelayanan", 1): ["baristanya ramah", "pelayanannya cepat", "kasirnya sopan dan sigap",
                       "pelayanan oke banget", "mbaknya ramah, ingat pesanan saya"],
    ("pelayanan", 0): ["pelayanannya lama banget", "baristanya jutek", "pesanan saya salah dibuat",
                       "antrinya lama, kasir cuma satu", "pegawainya sibuk main hp"],
    ("aplikasi", 1): ["aplikasinya gampang dipakai", "pesan lewat app praktis", "poinnya cepat masuk",
                      "fitur pesan ambil di tempat membantu"],
    ("aplikasi", 0): ["aplikasinya sering error pas bayar", "appnya lemot", "poin ga masuk-masuk",
                      "gagal login terus", "voucher ga bisa dipakai di aplikasi", "aplikasi force close"],
    ("antar", 1): ["pengirimannya cepat", "sampai masih dingin", "kemasannya rapi tidak tumpah"],
    ("antar", 0): ["pengirimannya lama", "kopinya tumpah di jalan", "esnya udah cair pas sampai",
                   "pesanan ga lengkap"],
    ("suasana", 1): ["tempatnya nyaman buat nugas", "wifinya kencang", "suasananya cozy",
                     "tempatnya bersih dan adem"],
    ("suasana", 0): ["tempatnya sempit", "ac nya ga dingin", "colokan susah", "berisik banget",
                     "mejanya kotor"],
}
PEMBUKA = ["", "", "", "overall ", "jujur ", "sejauh ini ", "", "pertama kali coba, "]
PENUTUP = {1: ["", "", " recommended!", " pasti balik lagi", " sukses terus", " top"],
           0: ["", "", " kecewa", " tolong diperbaiki", " ga lagi deh", " parah"],
           .5: ["", "", " biasa aja sih", " lumayan lah"]}


def buat_ulasan(rng, gerai, n=3000):
    topik = ["rasa", "harga", "pelayanan", "aplikasi", "antar", "suasana"]
    bobot = [.30, .15, .18, .17, .10, .10]
    baris = []
    for i in range(n):
        rating = int(rng.choice([1, 2, 3, 4, 5], p=[.10, .09, .14, .30, .37]))
        polar = 1 if rating >= 4 else (0 if rating <= 2 else .5)
        k = int(rng.choice([1, 2, 2, 3]))
        tps = list(rng.choice(topik, k, replace=False, p=bobot))
        bagian = []
        for j, tp in enumerate(tps):
            if polar == .5:
                pol = int(rng.random() < .5)
            else:
                pol = int(polar) if rng.random() < .88 else 1 - int(polar)
            bagian.append(rng.choice(FRASA[(tp, pol)]))
        teks = rng.choice(PEMBUKA) + ", ".join(bagian)
        sambung = rng.choice([", tapi ", " tapi ", ". "]) if polar == .5 and len(bagian) > 1 else None
        if sambung:
            teks = rng.choice(PEMBUKA) + bagian[0] + sambung + ", ".join(bagian[1:])
        teks += rng.choice(PENUTUP[polar])
        if rng.random() < .25:
            teks = teks.replace("banget", "bgt").replace("tidak", "gak").replace("sangat", "sgt")
        if rng.random() < .15:
            teks = teks.upper() if rating <= 2 else teks.capitalize()
        g = gerai.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        tgl = pd.Timestamp("2025-07-01") + pd.Timedelta(int(rng.integers(0, 273)), unit="D")
        baris.append({"id_ulasan": f"U{i + 1:04d}", "tanggal": tgl.date().isoformat(),
                      "id_gerai": g.id_gerai, "kanal": rng.choice(["aplikasi", "google maps"], p=[.7, .3]),
                      "rating": rating, "teks": teks.strip(), "topik": ";".join(tps)})
    df = pd.DataFrame(baris)
    df["sentimen"] = np.select([df.rating >= 4, df.rating <= 2], ["positif", "negatif"], "netral")
    return df


def main(folder="."):
    folder = Path(folder)
    rng = np.random.default_rng(BENIH)
    gerai = pd.read_csv(folder / "gerai.csv")
    pel = buat_pelanggan(rng, gerai)
    pel.to_csv(folder / "pelanggan.csv", index=False)
    kotori_pelanggan(rng, pel).to_csv(folder / "pelanggan_mentah.csv", index=False)
    buat_mutu_biji(rng).to_csv(folder / "mutu_biji.csv", index=False)
    buat_profil_rasa(rng).to_csv(folder / "profil_rasa.csv", index=False)
    buat_penjualan_harian(rng, gerai, folder).to_csv(folder / "penjualan_harian.csv", index=False)
    b, s = buat_sangrai(rng)
    b.to_csv(folder / "batch_sangrai.csv", index=False)
    s.to_csv(folder / "sensor_sangrai.csv", index=False)
    buat_audit_kasir(rng, gerai).to_csv(folder / "audit_kasir.csv", index=False)
    buat_sesi(rng).to_csv(folder / "sesi_aplikasi.csv", index=False)
    buat_ulasan(rng, gerai).to_csv(folder / "ulasan.csv", index=False)
    print("selesai")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
