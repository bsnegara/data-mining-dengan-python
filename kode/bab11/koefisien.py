import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA = "kode/data/"
df = pd.read_csv(DATA + "pelanggan.csv")
df["recency_10h"] = df["recency_hari"] / 10                      #<1>
df["tren"] = df["frekuensi_90h"] / (df["frekuensi_90h_lalu"] + 1)
df["promo_10pct"] = df["porsi_promo"] * 10
m = smf.logit("churn ~ recency_10h + tren + keluhan_90h + ragam_menu"
              " + promo_10pct + pakai_ulasan"
              " + C(tingkat, Treatment('perunggu'))", df).fit(disp=0)
tab = pd.DataFrame({"koef": m.params, "SE": m.bse, "p": m.pvalues,
                    "OR": np.exp(m.params),                      #<2>
                    "OR_bawah": np.exp(m.conf_int()[0]),
                    "OR_atas": np.exp(m.conf_int()[1])})
tab.index = [i.replace("C(tingkat, Treatment('perunggu'))[T.", "tingkat ")
             .replace("]", "") for i in tab.index]
print(tab.round(3).to_string())
print(f"pseudo-R2 McFadden = {m.prsquared:.3f}")                 #<3>
