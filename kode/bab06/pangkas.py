import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA = "kode/data/"
biji = pd.read_csv(DATA + "mutu_biji.csv")
fitur = ["kadar_air", "cacat_primer", "cacat_sekunder",
         "ukuran_screen", "densitas_gl", "skor_total",
         "ketinggian_mdpl", "lama_simpan_hari"]
Xl, Xu, yl, yu = train_test_split(biji[fitur], biji["keputusan"],
                                  test_size=.3, random_state=0,
                                  stratify=biji["keputusan"])
baris = []
for d in [1, 2, 3, 4, 5, 6, 8, 10, 12, None]:                    #<1>
    m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xl, yl)
    baris.append((d or 99, m.get_n_leaves(), m.score(Xl, yl),
                  m.score(Xu, yu)))
tab = pd.DataFrame(baris, columns=["kedalaman", "daun", "latih", "uji"])
print(tab.round(3).to_string(index=False))
tab.to_csv("gambar/bab06/kedalaman.dat", sep=" ", index=False)

jalur = DecisionTreeClassifier(random_state=0)                   #<2>
alfa = jalur.cost_complexity_pruning_path(Xl, yl).ccp_alphas
alfa = np.unique(np.round(alfa, 4))[::6]
skor = [cross_val_score(DecisionTreeClassifier(ccp_alpha=a,      #<3>
        random_state=0), Xl, yl, cv=5).mean() for a in alfa]
terbaik = alfa[int(np.argmax(skor))]
m = DecisionTreeClassifier(ccp_alpha=terbaik, random_state=0).fit(Xl, yl)
print(f"alfa terbaik {terbaik:.4f}: {m.get_n_leaves()} daun, "
      f"uji {m.score(Xu, yu):.3f}")
