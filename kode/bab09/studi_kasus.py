import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = "kode/data/"
biji = pd.read_csv(DATA + "mutu_biji.csv", parse_dates=["tanggal"])
rasa = ["aroma", "flavor", "aftertaste", "acidity", "body", "balance",
        "overall"]
latih = biji[biji["tanggal"] < "2025-01-01"]                     #<1>
uji = biji[biji["tanggal"] >= "2025-01-01"]
pipa = make_pipeline(StandardScaler(), KNeighborsClassifier())
kisi = {"kneighborsclassifier__n_neighbors": [5, 15, 25, 35, 51, 75],
        "kneighborsclassifier__weights": ["uniform", "distance"],
        "kneighborsclassifier__p": [1, 2]}                       #<2>
cari = GridSearchCV(pipa, kisi, cv=StratifiedKFold(
    5, shuffle=True, random_state=0)).fit(latih[rasa], latih["asal"])
print("terbaik:", {k.split("__")[1]: v
                   for k, v in cari.best_params_.items()},
      f"CV {cari.best_score_:.3f}")
pred = cari.predict(uji[rasa])
print(f"akurasi uji 2025: {(pred == uji['asal']).mean():.3f}\n")
urutan = ["Gayo", "Kerinci", "Toraja", "Flores"]
print(pd.DataFrame(confusion_matrix(uji["asal"], pred, labels=urutan),
                   index=urutan, columns=urutan).to_string())    #<3>
print(classification_report(uji["asal"], pred, digits=3))

p = cari.predict_proba(uji[rasa]).max(1)                         #<4>
print("ambang suara  cakupan  akurasi")
for t in [0.0, 0.5, 0.6, 0.7, 0.8]:
    pilih = p >= t
    print(f"    {t:.1f}       {pilih.mean():.3f}    "
          f"{(pred[pilih] == uji['asal'].to_numpy()[pilih]).mean():.3f}")

model = cari.best_estimator_                                     #<5>
satu = uji[uji["asal"] == "Kerinci"].iloc[[3]]
jarak, idx = model[-1].kneighbors(model[0].transform(satu[rasa]), 5)
tetangga = latih.iloc[idx[0]][["id_kiriman", "asal", "acidity", "body"]]
print(f"\nkiriman {satu['id_kiriman'].item()} (sebenarnya Kerinci),"
      f" lima tetangga terdekat:")
print(tetangga.assign(jarak=jarak[0].round(2)).to_string(index=False))
