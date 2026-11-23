import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = "kode/data/"
df = pd.read_csv(DATA + "pelanggan.csv")
fitur = ["frekuensi_90h", "recency_hari", "porsi_promo", "keluhan_90h"]
model = make_pipeline(StandardScaler(), LogisticRegression())

auc = cross_val_score(model, df[fitur], df["churn"], cv=5,
                      scoring="roc_auc").mean()
print(f"fitur sah          : AUC = {auc:.3f}")

df["belanja_depan_ada"] = (df["belanja_bulan_depan"] > 0).astype(int)  #<1>
bocor = fitur + ["belanja_depan_ada"]
auc = cross_val_score(model, df[bocor], df["churn"], cv=5,
                      scoring="roc_auc").mean()
print(f"dengan fitur bocor : AUC = {auc:.3f}")
