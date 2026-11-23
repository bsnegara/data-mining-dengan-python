import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA = "kode/data/"
trx = pd.read_csv(DATA + "transaksi_lengkap_2026q1.csv",
                  parse_dates=["waktu"])
harian = trx.set_index("waktu")["nilai"].resample("D").sum()  #<1>

fig, ax = plt.subplots(figsize=(8, 3))                    #<2>
(harian / 1e6).plot(ax=ax)
ax.axvspan("2026-02-18", "2026-03-19", alpha=0.15)        #<3>
ax.set_ylabel("omzet (juta Rp)")
fig.savefig("omzet_harian.png", dpi=150, bbox_inches="tight")

trx["hari"] = trx["waktu"].dt.dayofweek                   #<4>
trx["jam"] = trx["waktu"].dt.hour
biasa = trx[trx["waktu"] < "2026-02-18"]
peta = biasa.pivot_table(index="hari", columns="jam",
                         values="no_struk", aggfunc="nunique")
fig, ax = plt.subplots(figsize=(8, 3))
sns.heatmap(peta, cmap="YlOrBr", ax=ax)                   #<5>
fig.savefig("peta_jam_hari.png", dpi=150, bbox_inches="tight")
print((harian / 1e6).describe().round(2).to_string())
