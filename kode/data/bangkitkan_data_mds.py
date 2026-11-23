"""Membangkitkan kemiripan_panel.csv untuk Bab 21 (penskalaan multidimensi).

Lima anggota panel cita rasa menilai setiap pasangan dari 24 kopi asal
dengan skor beda 1 (sangat mirip) sampai 9 (sangat berbeda). Penilaian
berasal dari koordinat laten yang sama dengan profil_rasa.csv (intensitas,
kecerahan, manis), dilewatkan fungsi monoton tak linier, ditambah derau
setiap panelis, lalu dibulatkan.

Pemakaian:  python bangkitkan_data_mds.py [folder]
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bangkitkan_data_lanjut import PROFIL  # noqa: E402

BENIH = 2027 + 21
PANELIS = ["Made", "Ayu", "Rizal", "Tiur", "Yosef"]
BOBOT = {  # setiap panelis menimbang dimensi secara sedikit berbeda
    "Made": (1.0, 1.0, 1.0), "Ayu": (.8, 1.2, 1.3), "Rizal": (1.2, .9, .7),
    "Tiur": (1.0, 1.1, 1.1), "Yosef": (1.1, .8, 1.0)}


def main(folder):
    rng = np.random.default_rng(BENIH)
    nama = list(PROFIL)
    Z = np.array([PROFIL[k] for k in nama])
    baris = []
    for a, b in itertools.combinations(range(len(nama)), 2):
        skor = []
        for p in PANELIS:
            w = np.array(BOBOT[p])
            d = np.sqrt((w * (Z[a] - Z[b]) ** 2).sum())
            s = 1 + 8 * (1 - np.exp(-d / 1.1)) + rng.normal(0, .6)
            skor.append(int(np.clip(round(s), 1, 9)))
        baris.append({"kopi_a": nama[a], "kopi_b": nama[b],
                      **{f"skor_{p.lower()}": s for p, s in zip(PANELIS, skor)},
                      "skor_beda": round(float(np.mean(skor)), 1)})
    df = pd.DataFrame(baris)
    df.to_csv(Path(folder) / "kemiripan_panel.csv", index=False)
    print(f"kemiripan_panel.csv: {len(df)} pasangan, {len(PANELIS)} panelis")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
