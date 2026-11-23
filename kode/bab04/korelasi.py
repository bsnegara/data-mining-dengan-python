import numpy as np
import pandas as pd
from scipy import stats

DATA = "kode/data/"
pel = pd.read_csv(DATA + "pelanggan.csv")
kol = ["frekuensi_90h", "recency_hari", "nilai_90h", "porsi_promo",
       "keluhan_90h"]
singkat = dict(zip(kol, ["frek", "recency", "nilai", "promo", "keluhan"]))
print(pel[kol].rename(columns=singkat).corr().round(2).to_string())  #<1>
print(pel[kol].corr(method="spearman")                            #<2>
      .loc["frekuensi_90h", "recency_hari"].round(2))

print(pel.groupby("churn")[["recency_hari", "porsi_promo"]]       #<3>
      .mean().round(2).to_string())

tab = pd.crosstab(pel["tingkat"], pel["churn"])                   #<4>
print(tab.to_string())
chi2, p, dof, _ = stats.chi2_contingency(tab)
n = tab.values.sum()
v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))                    #<5>
print(f"chi2={chi2:.1f}  dof={dof}  p={p:.1e}  V={v:.2f}")
