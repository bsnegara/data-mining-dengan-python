import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text

DATA = "kode/data/"
biji = pd.read_csv(DATA + "mutu_biji.csv", parse_dates=["tanggal"])
fitur = ["kadar_air", "cacat_primer", "cacat_sekunder",
         "ukuran_screen", "densitas_gl", "skor_total"]
latih = biji[biji["tanggal"] < "2025-01-01"]                     #<1>
uji = biji[biji["tanggal"] >= "2025-01-01"]
print("latih:", len(latih), " uji:", len(uji))

lama = ((uji["kadar_air"] <= 12.5) & (uji["cacat_primer"] <= 5)) #<2>
pred_lama = lama.map({True: "terima", False: "tolak"})
print("aturan lama, kecocokan:",
      round(accuracy_score(uji["keputusan"], pred_lama), 3))

cari = GridSearchCV(                                             #<3>
    DecisionTreeClassifier(random_state=0),
    {"max_leaf_nodes": [4, 6, 8, 10, 12, 16, 24],
     "min_samples_leaf": [10, 20, 40]}, cv=5)
cari.fit(latih[fitur], latih["keputusan"])
print("terbaik:", cari.best_params_,
      "CV:", round(cari.best_score_, 3))
print("pohon terbaik, kecocokan uji:",
      round(cari.score(uji[fitur], uji["keputusan"]), 3))

pohon = DecisionTreeClassifier(max_leaf_nodes=8,                 #<4>
                               min_samples_leaf=20, random_state=0)
pohon.fit(latih[fitur], latih["keputusan"])
pred = pohon.predict(uji[fitur])
print("pohon 8 daun, kecocokan uji:",
      round(accuracy_score(uji["keputusan"], pred), 3))
print(confusion_matrix(uji["keputusan"], pred,                   #<5>
                       labels=["terima", "tolak"]))
print(export_text(pohon, feature_names=fitur, decimals=1))
penting = pd.Series(pohon.feature_importances_, index=fitur)
print(penting.sort_values(ascending=False).round(3).to_string()) #<6>
