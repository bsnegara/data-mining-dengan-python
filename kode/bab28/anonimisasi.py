import hashlib
import hmac
import numpy as np
import pandas as pd

d = pd.read_csv("kode/data/pelanggan.csv")
kuasi = ["umur", "jenis_kelamin", "gerai_utama",                #<1>
         "tanggal_daftar"]


def ukuran_kelas(df, kolom):
    grup = df.groupby(kolom, observed=True)
    return grup[kolom[0]].transform("size")


k = ukuran_kelas(d, kuasi)
print(f"anggota yang unik (k = 1): {(k == 1).mean():.1%}")

g = d.copy()                                                     #<2>
g["umur"] = pd.cut(g.umur, [16, 24, 34, 44, 54, 99],
                   labels=["17-24", "25-34", "35-44", "45-54", "55+"])
g["tanggal_daftar"] = g.tanggal_daftar.str[:4]          # tahun saja
g["gerai_utama"] = g.kota                               # kota saja
for nama, kolom in [("umur, jk, kota, tahun", kuasi),
                    ("umur, jk, kota", kuasi[:3])]:
    k = ukuran_kelas(g, kolom)
    print(f"generalisasi ({nama}): k minimum {k.min()}, "
          f"k < 5: {(k < 5).mean():.1%}")
aman = g[k >= 5]
print(f"setelah {len(g) - len(aman)} baris dengan k < 5 dihapus: "
      f"k minimum {ukuran_kelas(aman, kuasi[:3]).min()}")

KUNCI = b"simpan-di-brankas-rahasia"                             #<3>


def pseudonim(x):
    kode = hmac.new(KUNCI, x.encode(), hashlib.sha256)
    return kode.hexdigest()[:12]


print("\npseudonim:", d.id_anggota[:3].map(pseudonim).tolist())
polos = hashlib.sha256(b"A00001").hexdigest()[:12]
tebak = {hashlib.sha256(f"A{i:05d}".encode()).hexdigest()[:12]:
         f"A{i:05d}" for i in range(1, 100000)}
print(f"hash tanpa kunci {polos} dibalik: {tebak[polos]}")

rng = np.random.default_rng(28)                                  #<4>
benar = int((d.kota == "Batam").sum())
for eps in (0.1, 1.0):
    jawab = benar + rng.laplace(0, 1 / eps, 5)
    print(f"jumlah anggota Batam (epsilon {eps}): "
          f"{np.round(jawab).astype(int).tolist()} (benar {benar})")
