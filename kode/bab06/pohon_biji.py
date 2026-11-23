import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

DATA = "kode/data/"
biji = pd.read_csv(DATA + "mutu_biji.csv")
fitur = ["kadar_air", "cacat_primer", "cacat_sekunder",
         "ukuran_screen", "densitas_gl", "skor_total"]
X, y = biji[fitur], biji["keputusan"]

pohon = DecisionTreeClassifier(criterion="entropy",              #<1>
                               max_depth=3, random_state=0)
pohon.fit(X, y)
print(export_text(pohon, feature_names=fitur,                    #<2>
                  show_weights=True, decimals=1))
print("akurasi latih:", round(pohon.score(X, y), 3))
