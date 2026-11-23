import numpy as np
import pandas as pd

a = pd.read_csv("kode/data/audit_kasir.csv", parse_dates=["tanggal"])
MULAI = pd.Timestamp("2026-02-01")
jan = a[a.tanggal < MULAI]
P_SEMUA = jan.struk_batal.sum() / jan.jumlah_struk.sum()

def pantau(kid, k=1.0, h=5.0, lam=0.1, L=3.0):                   #<1>
    s = a[a.id_kasir == kid].set_index("tanggal")
    f1 = s[s.index < MULAI]                            # fase I
    p0 = (f1.struk_batal.sum() + 1000 * P_SEMUA) \
        / (f1.jumlah_struk.sum() + 1000)                         #<2>
    s = s[s.index >= MULAI]
    n = s.jumlah_struk
    z = (s.struk_batal - n * p0) / np.sqrt(n * p0 * (1 - p0))
    c = e = 0.0
    batas_e = L * np.sqrt(lam / (2 - lam))
    sinyal, jejak = {}, []
    for hari, zt in zip((s.index - MULAI).days + 1, z.to_numpy()):
        c = max(0.0, c + zt - k)                                 #<3>
        e = lam * zt + (1 - lam) * e                             #<4>
        jejak.append((hari, zt, c, e))
        for nama, ya in [("Shewhart", zt > 3), ("CUSUM", c > h),
                         ("EWMA", e > batas_e)]:
            if ya and nama not in sinyal:
                sinyal[nama] = hari
    return sinyal, jejak

print(f"p batal rata-rata Januari: {P_SEMUA:.4f}")
print("hari ke- sinyal pertama sejak 1 Februari:")
for kid in ["G07-K2", "G31-K3"]:                                 #<5>
    sinyal, jejak = pantau(kid)
    teks = [f"{n} {sinyal.get(n, '-')}"
            for n in ["Shewhart", "CUSUM", "EWMA"]]
    print(f"  {kid}: " + ", ".join(teks))
    if kid == "G07-K2":
        with open("gambar/bab22/cusum.dat", "w") as f:
            f.write("hari z c e\n")
            for hari, zt, c, e in jejak:
                f.write(f"{hari} {zt:.3f} {c:.3f} {e:.3f}\n")

alarm = {"Shewhart": 0, "CUSUM": 0, "EWMA": 0}                   #<6>
jujur = [k for k in a.id_kasir.unique()
         if k not in ("G07-K2", "G31-K3", "G18-K1")]
for kid in jujur:
    for nama in pantau(kid)[0]:
        alarm[nama] += 1
print(f"kasir jujur yang memicu alarm Feb-Mar (dari {len(jujur)}):")
print("  " + ", ".join(f"{n} {v}" for n, v in alarm.items()))
