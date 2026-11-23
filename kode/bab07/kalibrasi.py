import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from siapkan import muat, pipa

X, y = muat()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                  random_state=7)
model = {"logistik": pipa(LogisticRegression(max_iter=2000)),
         "hutan": pipa(RandomForestClassifier(300, min_samples_leaf=5,
                       random_state=0, n_jobs=-1), False),
         "nb": pipa(GaussianNB())}
keluar = {}
for nama, m in model.items():
    p = m.fit(Xl, yl).predict_proba(Xu)[:, 1]
    nyata, ramal = calibration_curve(yu, p, n_bins=8,            #<1>
                                     strategy="quantile")
    keluar[nama] = pd.DataFrame({"ramal": ramal, "nyata": nyata})
    print(f"{nama:9s} Brier {brier_score_loss(yu, p):.4f}  "     #<2>
          f"rata-rata ramalan {p.mean():.3f}")
print(f"porsi churn uji {yu.mean():.3f}")
for nama, d in keluar.items():
    d.round(4).to_csv(f"gambar/bab07/kalibrasi-{nama}.dat", sep=" ",
                      index=False)
