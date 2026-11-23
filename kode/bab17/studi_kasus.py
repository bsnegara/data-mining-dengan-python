import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from siapkan import keranjang, muat_struk

pd.set_option("display.width", 120)
t = muat_struk()
B = keranjang(t, konteks=True, minimal=2)                        #<1>
f = fpgrowth(B, min_support=0.002, use_colnames=True)
r = association_rules(f, metric="lift", min_threshold=1.1)       #<2>
r = r[r["antecedents"].map(lambda s: any(i.startswith("[") for i in s))]
r = r.assign(jika=r["antecedents"].map(lambda s: ", ".join(sorted(s))),
             maka=r["consequents"].map(lambda s: ", ".join(sorted(s))))
print(f"{len(B)} struk berisi >= 2 item; {len(r)} aturan berkonteks:")
print(r.sort_values("lift", ascending=False)[
    ["jika", "maka", "support", "confidence", "lift"]].head(12)
    .round(3).to_string(index=False))

pagi = B["[pagi]"]                                               #<3>
a, c = B["Americano"], B["Croissant Mentega"]
def lift(m):
    return (a[m] & c[m]).mean() / (a[m].mean() * c[m].mean())
print(f"\nlift Americano & Croissant: semua {lift(pagi | ~pagi):.2f}, "
      f"pagi {lift(pagi):.2f}, bukan pagi {lift(~pagi):.2f}")

g = (B[B["Kopi Susu Gula Aren"]].drop(columns=["[pagi]", "[berbuka]",
                                               "Kopi Susu Gula Aren"])
     .mean().sort_values(ascending=False))                       #<4>
dasar = B[~B["Kopi Susu Gula Aren"]].drop(
    columns=["[pagi]", "[berbuka]", "Kopi Susu Gula Aren"]).mean()
print("\nteman Kopi Susu Gula Aren (struk >= 2 item):")
print(pd.DataFrame({"bersama gula aren": g, "tanpa gula aren": dasar[g.index],
                    "rasio": g / dasar[g.index]}).head(6).round(3)
      .to_string())
semua = association_rules(fpgrowth(B, min_support=0.002, use_colnames=True),
                          metric="lift", min_threshold=0)
semua[["support", "confidence", "lift"]].round(4).to_csv(        #<5>
    "gambar/bab17/aturan.dat", sep=" ", index=False)
