import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from siapkan import muat, pra

df, X, y = muat()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                  random_state=7)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
model = {"logistik": LogisticRegression(C=1, max_iter=2000),
         "SVM linier": SVC(kernel="linear", C=0.1),
         "SVM RBF": SVC(kernel="rbf", C=10, gamma=0.01),          #<1>
         "random forest": RandomForestClassifier(500, random_state=0)}
print("model           AUC-CV  AUC-uji")
for n, m in model.items():
    p = make_pipeline(pra(), m)
    s = cross_val_score(p, Xl, yl, cv=cv, scoring="roc_auc").mean()
    p.fit(Xl, yl)
    skor = (p.decision_function(Xu) if hasattr(p, "decision_function")
            else p.predict_proba(Xu)[:, 1])
    print(f"{n:15s} {s:7.3f} {roc_auc_score(yu, skor):8.3f}")

svm = make_pipeline(pra(), CalibratedClassifierCV(                #<2>
    SVC(kernel="linear", C=0.1), cv=5, method="sigmoid"))
svm.fit(Xl, yl)
p = svm.predict_proba(Xu)[:, 1]
yv = yu.to_numpy()
print("\natas  bawah  otomatis  presisi-spec  presisi-kom  cupping")
for atas, bawah in [(0.5, 0.5), (0.8, 0.2), (0.9, 0.1), (0.95, 0.05),
                    (0.95, 0.2)]:
    spec, kom = p >= atas, p <= bawah                            #<3>
    print(f"{atas:4.2f} {bawah:6.2f} {(spec | kom).mean():9.1%} "
          f"{yv[spec].mean():13.3f} {1 - yv[kom].mean():12.3f} "
          f"{1 - (spec | kom).mean():8.1%}")

otomatis = ((p >= 0.95) | (p <= 0.2)).mean()                     #<4>
hemat = otomatis * 500 * 500_000
print(f"\n~500 kiriman/tahun, cupping Rp500 ribu/kiriman; ambang 0,95/0,2"
      f"\nmenghemat ~Rp{hemat / 1e6:.0f} jt per tahun")
