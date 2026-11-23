import joblib
import numpy as np
import pandas as pd
from siapkan import BIAYA, DATA, NILAI_TAHAN, fitur

model = joblib.load("kode/bab27/model/churn_v1.joblib")
d = pd.read_csv(DATA + "pelanggan_2026q3.csv")
d["p"] = model.predict_proba(fitur(d, "2026-09-30"))[:, 1]
v = d.kelompok == "voucer"

c = d.groupby("kelompok").churn.agg(["size", "mean"])            #<1>
print(c.round(4).to_string())
beda = c.loc["kontrol", "mean"] - c.loc["voucer", "mean"]
se = np.sqrt(sum(c["mean"] * (1 - c["mean"]) / c["size"]))
print(f"efek voucer: churn turun {beda:.3f} "
      f"(95%: {beda - 1.96 * se:.3f} sampai {beda + 1.96 * se:.3f})")

d["desil"] = pd.qcut(d.p, 10, labels=range(1, 11))              #<2>
t = d.pivot_table(index="desil", columns="kelompok", values="churn",
                  observed=True)
t["uplift"] = t.kontrol - t.voucer
t["laba_per_org"] = (t.uplift * NILAI_TAHAN - BIAYA).round(-2)
t["laba_per_org"] = t.laba_per_org.astype(int)
print("\nchurn per desil skor model:")
print(t.to_string(float_format="%.3f"))
t.assign(d=range(1, 11)).to_csv("gambar/bab27/uplift.dat", sep=" ",
                                index=False, float_format="%.4f")


def nilai_kebijakan(sasar, B=500, seed=0):                       #<3>
    """Laba per 1.000 anggota bila voucer dikirim ke anggota 'sasar',
    ditaksir dari uji acak (bootstrap untuk selang)."""
    rng = np.random.default_rng(seed)
    hasil = []
    for b in range(B + 1):
        i = (np.arange(len(d)) if b == 0 else
             rng.integers(0, len(d), len(d)))
        s, vv, y = sasar[i], v.to_numpy()[i], d.churn.to_numpy()[i]
        up = y[s & ~vv].mean() - y[s & vv].mean()
        hasil.append(s.mean() * 1000 * (up * NILAI_TAHAN - BIAYA))
    return hasil[0], np.percentile(hasil[1:], [2.5, 97.5])


peka = d.porsi_promo >= d.porsi_promo.median()
ambang = BIAYA / (0.3 * NILAI_TAHAN)
kebijakan = {"semua anggota": np.ones(len(d), bool),
             "20% skor tertinggi": (d.p >= d.p.quantile(.8))
             .to_numpy(),
             f"p >= {ambang:.3f} (Bab 7)": (d.p >= ambang).to_numpy(),
             "p >= ambang & peka promo": ((d.p >= ambang)
                                          & peka).to_numpy()}
print("\nlaba program per 1.000 anggota (ribu rupiah):")
for nama, s in kebijakan.items():
    m, (lo, hi) = nilai_kebijakan(s)
    print(f"  {nama:26s} sasaran {s.mean():5.1%}  {m / 1e3:8.0f}"
          f"  [{lo / 1e3:6.0f}, {hi / 1e3:6.0f}]")
