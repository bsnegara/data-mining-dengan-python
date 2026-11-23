import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

y = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])                     #<1>
s = np.array([.95, .85, .80, .70, .55, .45, .40, .30, .20, .10])
fpr, tpr, ambang = roc_curve(y, s, drop_intermediate=False)
for a, f, t in zip(ambang[1:], fpr[1:], tpr[1:]):
    print(f"ambang {a:.2f}: FPR {f:.3f}  TPR {t:.3f}")
pos, neg = s[y == 1], s[y == 0]                                  #<2>
pasangan = (pos[:, None] > neg[None, :]).mean()
print("AUC (pasangan) =", pasangan, " AUC (sklearn) =",
      roc_auc_score(y, s))
