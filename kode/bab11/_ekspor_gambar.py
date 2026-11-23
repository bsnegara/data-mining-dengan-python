import numpy as np
import pandas as pd

z = np.linspace(-6, 6, 121)
pd.DataFrame({"z": z, "p": 1 / (1 + np.exp(-z))}).round(4).to_csv(
    "gambar/bab11/sigmoid.dat", sep=" ", index=False)
df = pd.read_csv("kode/data/pelanggan.csv")
b = pd.cut(df["recency_hari"], [-1, 2, 5, 8, 12, 17, 25, 35, 50, 70, 200])
g = df.groupby(b, observed=True).agg(x=("recency_hari", "mean"),
                                     p=("churn", "mean"))
g["logit"] = np.log(g["p"] / (1 - g["p"]))
g.round(4).to_csv("gambar/bab11/recency_logit.dat", sep=" ", index=False)
