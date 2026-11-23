"""Pembangkit dataset sintetis Kopi Nusantara, bagian ketiga (Bab 18).

Keluaran: riwayat_aplikasi_2025.csv, riwayat pesanan aplikasi anggota
loyalitas sepanjang 2025 (satu baris per menu per pesanan). Selera setiap
anggota dibangun dari segmen latennya di pelanggan.csv, tiga faktor rasa
laten (manis-susu, pahit-hitam, segar-nonkopi), dan dua sampai tiga menu
favorit pribadi. Seluruh data REKAAN dengan benih acak tetap.

Pemakaian:  python bangkitkan_data_rekomendasi.py [folder]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENIH = 2027 + 18
#            manis-susu  pahit-hitam  segar-nonkopi  makanan
FAKTOR_MENU = {
    "K01": (1.0, 0.0, 0.1, 0.0), "K02": (0.9, 0.0, 0.2, 0.0),
    "K03": (0.9, 0.0, 0.1, 0.1), "K04": (0.6, 0.3, 0.0, 0.0),
    "K05": (0.5, 0.4, 0.0, 0.0), "K06": (0.8, 0.0, 0.4, 0.0),
    "H01": (0.0, 0.9, 0.1, 0.0), "H02": (0.0, 1.0, 0.0, 0.0),
    "H03": (0.0, 1.0, 0.0, 0.0), "H04": (0.0, 1.0, 0.1, 0.0),
    "H05": (0.1, 0.8, 0.0, 0.0), "N01": (0.5, 0.0, 0.5, 0.0),
    "N02": (0.6, 0.0, 0.4, 0.0), "N03": (0.4, 0.0, 0.7, 0.0),
    "N04": (0.0, 0.0, 1.0, 0.0), "N05": (0.0, 0.0, 1.0, 0.0),
    "M01": (0.2, 0.2, 0.0, 1.0), "M02": (0.2, 0.0, 0.1, 1.0),
    "M03": (0.0, 0.3, 0.0, 1.0), "M04": (0.0, 0.0, 0.2, 1.0),
    "M05": (0.3, 0.0, 0.0, 1.0), "M06": (0.3, 0.0, 0.0, 1.0)}
SELERA_SEGMEN = {          # bobot faktor rata-rata tiap segmen laten
    "mahasiswa": (1.4, 0.2, 1.0, 0.5), "pekerja": (0.9, 0.9, 0.2, 0.6),
    "keluarga": (0.8, 0.2, 1.1, 1.3), "penikmat": (0.2, 2.0, 0.1, 0.4)}


def main(folder="."):
    folder = Path(folder)
    rng = np.random.default_rng(BENIH)
    menu = pd.read_csv(folder / "menu.csv")
    kode = menu["kode_menu"].to_numpy()
    F = np.array([FAKTOR_MENU[k] for k in kode])
    populer = np.log(1 + menu["kode_menu"].map(
        {"K01": 6, "K02": 3, "H01": 3, "K04": 2.5}).fillna(1.5).to_numpy())
    agt = pd.read_csv(folder / "pelanggan.csv")
    hari = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    baris, no = [], 0
    for _, a in agt.iterrows():
        n = rng.poisson(a["frekuensi_90h"] * 4 * a["porsi_aplikasi"] * 0.5)
        if n == 0:
            continue
        u = np.array(SELERA_SEGMEN[a["segmen_laten"]]) * \
            rng.lognormal(0, 0.35, 4)
        skor = F @ u + 0.6 * populer + rng.normal(0, 0.3, len(kode))
        fav = rng.choice(len(kode), rng.integers(2, 4), replace=False,
                         p=np.exp(skor) / np.exp(skor).sum())
        skor[fav] += rng.uniform(1.0, 2.0, len(fav))
        p = np.exp(1.3 * skor)
        p /= p.sum()
        tgl = np.sort(rng.choice(hari, n))
        for t in tgl:
            no += 1
            k = 1 + (rng.random() < 0.3)
            for j in rng.choice(len(kode), k, replace=False, p=p):
                baris.append((f"P{no:06d}", pd.Timestamp(t).date(),
                              a["id_anggota"], kode[j], 1))
    df = pd.DataFrame(baris, columns=["id_pesanan", "tanggal", "id_anggota",
                                      "kode_menu", "qty"])
    df.to_csv(folder / "riwayat_aplikasi_2025.csv", index=False)
    print(f"riwayat_aplikasi_2025.csv: {len(df)} baris, "
          f"{df['id_pesanan'].nunique()} pesanan, "
          f"{df['id_anggota'].nunique()} anggota")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
