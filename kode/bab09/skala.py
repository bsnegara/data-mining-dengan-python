import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATA = "kode/data/"
biji = pd.read_csv(DATA + "mutu_biji.csv")
rasa = ["aroma", "flavor", "aftertaste", "acidity", "body", "balance",
        "overall"]
fisik = ["kadar_air", "cacat_primer", "cacat_sekunder", "ukuran_screen",
         "densitas_gl"]
print(biji[fisik + ["acidity"]].agg(["min", "max", "std"]).round(2)
      .to_string())                                              #<1>
cv = StratifiedKFold(5, shuffle=True, random_state=0)
print("\natribut          tanpa skala  min-maks  skor-z")
for nama, kol in [("rasa (7)", rasa), ("fisik (5)", fisik),
                  ("rasa + fisik", rasa + fisik)]:
    hasil = []
    for skala in [None, MinMaxScaler(), StandardScaler()]:       #<2>
        m = KNeighborsClassifier(25)
        if skala is not None:
            m = make_pipeline(skala, m)
        hasil.append(cross_val_score(m, biji[kol], biji["asal"],
                                     cv=cv).mean())
    print(f"{nama:15s}  {hasil[0]:.3f}       {hasil[1]:.3f}"
          f"     {hasil[2]:.3f}")
