"""Membangkitkan pelanggan_2026q3.csv untuk Bab 27 (proyek ujung ke ujung).

Potret 8.000 anggota aktif per 30 September 2026 dengan atribut yang sama
seperti pelanggan.csv. Dua pergeseran disengaja: pesaing gencar berpromosi
sehingga porsi transaksi promo naik (terutama mahasiswa dan pekerja), dan
porsi pesanan lewat aplikasi naik. Anggota dibagi acak menjadi kelompok
"voucer" dan "kontrol" untuk uji coba program retensi. Voucer menahan
sebagian anggota yang semula akan berhenti; peluangnya lebih besar bagi
anggota yang peka promo. Kolom churn adalah hasil yang teramati sesudah
uji coba (Oktober--Desember 2026).

Pemakaian:  python bangkitkan_data_proyek.py [folder]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import bangkitkan_data_lanjut as L  # noqa: E402

BENIH = 2027 + 27


def main(folder):
    folder = Path(folder)
    rng = np.random.default_rng(BENIH)
    L.POTONG = pd.Timestamp("2026-09-30")
    geser = {"mahasiswa": (1.30, 1.15), "pekerja": (1.35, 1.20),
             "keluarga": (1.10, 1.20), "penikmat": (1.00, 1.10)}
    L.SEGMEN = {k: v[:8] + (min(.95, v[8] * geser[k][1]), v[9] * geser[k][0])
                for k, v in L.SEGMEN.items()}
    gerai = pd.read_csv(folder / "gerai.csv")
    df = L.buat_pelanggan(rng, gerai)
    df["id_anggota"] = [f"A{int(x[1:]) + 10000:05d}" for x in df.id_anggota]
    df["kelompok"] = rng.choice(["voucer", "kontrol"], len(df))
    p_tahan = np.clip(.10 + .60 * df.porsi_promo, 0, .8)
    tahan = (df.kelompok == "voucer") & (df.churn == 1) & (rng.random(len(df)) < p_tahan)
    df.loc[tahan, "churn"] = 0
    df.loc[tahan, "belanja_bulan_depan"] = (df.nilai_90h[tahan] / 3 * .8).round(-2).astype(int)
    df.to_csv(folder / "pelanggan_2026q3.csv", index=False)
    print("selesai:", len(df), "anggota; churn", round(df.churn.mean(), 3))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
