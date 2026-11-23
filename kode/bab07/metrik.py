import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from siapkan import muat, pipa

X, y = muat()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                  random_state=7)
m = pipa(LogisticRegression(max_iter=2000)).fit(Xl, yl)
p = m.predict_proba(Xu)[:, 1]                                    #<1>
yp = (p >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(yu, yp).ravel()                #<2>
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"akurasi  {(tp + tn) / len(yu):.3f}")
print(f"presisi  {precision_score(yu, yp):.3f}")
print(f"recall   {recall_score(yu, yp):.3f}")
print(f"spesifik {tn / (tn + fp):.3f}")
print(f"F1       {f1_score(yu, yp):.3f}")
print(f"bal.akur {balanced_accuracy_score(yu, yp):.3f}")
print(f"MCC      {matthews_corrcoef(yu, yp):.3f}")

print("\nambang  presisi  recall  F1     ditandai")              #<3>
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    yp = (p >= t).astype(int)
    print(f" {t:.1f}    {precision_score(yu, yp):.3f}    "
          f"{recall_score(yu, yp):.3f}   {f1_score(yu, yp):.3f}  "
          f"{yp.sum():5d}")
