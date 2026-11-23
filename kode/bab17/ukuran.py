import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from siapkan import keranjang, muat_struk

pd.set_option("display.width", 120)
t = muat_struk()
for nama, minimal in [("semua struk", 1), ("struk >= 2 item", 2)]:    #<1>
    B = keranjang(t, minimal=minimal)
    f = fpgrowth(B, min_support=0.002, use_colnames=True)
    r = association_rules(f, metric="lift", min_threshold=0)
    g = r[r["antecedents"] == frozenset({"Kopi Susu Gula Aren"})]
    g = g.assign(konsekuen=g["consequents"].map(lambda s: ", ".join(s)))
    print(f"{nama}: {len(B)} struk; lift semua aturan "
          f"{r['lift'].min():.2f}-{r['lift'].max():.2f}")
    print(g.sort_values("confidence", ascending=False)[                #<2>
        ["konsekuen", "support", "confidence", "lift", "kulczynski"]]
        .head(5).round(3).to_string(index=False))
    print()
