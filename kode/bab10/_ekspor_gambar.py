import sys
sys.path.insert(0, "kode/bab10")
from gerai_tahunan import matang  # noqa: E402

for t in ["kampus", "perkantoran", "mal", "permukiman"]:
    matang[matang["tipe_lokasi"] == t][["luas_m2", "omzet"]].to_csv(
        f"gambar/bab10/luas_{t}.dat", sep=" ", index=False)
