import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = "kode/data/"
biji = pd.read_csv(DATA + "mutu_biji.csv")
rasa = ["aroma", "flavor", "aftertaste", "acidity", "body", "balance",
        "overall"]
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))  #<1>
cv = StratifiedKFold(5, shuffle=True, random_state=0)
print(f"akurasi CV softmax: "
      f"{cross_val_score(m, biji[rasa], biji['asal'], cv=cv).mean():.3f}")
m.fit(biji[rasa], biji["asal"])
koef = pd.DataFrame(m[-1].coef_, index=m[-1].classes_, columns=rasa)  #<2>
print(koef.round(2).to_string())
contoh = biji[rasa].iloc[[0]]
p = m.predict_proba(contoh)[0]                                   #<3>
print("kiriman pertama:", ", ".join(f"{k} {v:.3f}"
                                    for k, v in zip(m.classes_, p)),
      "| asal sebenarnya:", biji["asal"].iloc[0])
