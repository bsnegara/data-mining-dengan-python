import pandas as pd
from sklearn.tree import DecisionTreeClassifier

biji = pd.read_csv("kode/data/mutu_biji.csv")
m = DecisionTreeClassifier(max_depth=2, criterion="entropy").fit(
    biji[["kadar_air", "skor_total"]], biji["keputusan"])
t = m.tree_
print("ambang:", [(int(f), round(float(v), 2)) for f, v in zip(t.feature, t.threshold) if f >= 0])
s = biji.sample(450, random_state=3)
s.assign(kelas=(s.keputusan == "tolak").astype(int))[["kadar_air", "skor_total", "kelas"]].to_csv(
    "gambar/bab06/sebar_biji.dat", sep=" ", index=False)
