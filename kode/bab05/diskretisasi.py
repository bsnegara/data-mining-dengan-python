import pandas as pd
from sklearn.tree import DecisionTreeClassifier

DATA = "kode/data/"
df = pd.read_csv(DATA + "pelanggan_bersih.csv")
r = df["recency_hari"]

lebar = pd.cut(r, bins=4)                                        #<1>
frek = pd.qcut(r, q=4, duplicates="drop")                        #<2>
print(lebar.value_counts(sort=False).to_string())
print(frek.value_counts(sort=False).to_string())

pohon = DecisionTreeClassifier(max_leaf_nodes=4,                 #<3>
                               min_samples_leaf=300)
pohon.fit(df[["recency_hari"]], df["churn"])
ambang = sorted(t for t in pohon.tree_.threshold if t > 0)
print("ambang terbimbing:", [round(float(t), 1) for t in ambang])
kelas = pd.cut(r, [-1] + ambang + [r.max()])
print(df.groupby(kelas, observed=True)["churn"]                  #<4>
      .agg(["size", "mean"]).round(3).to_string())
