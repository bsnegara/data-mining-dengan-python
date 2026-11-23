import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from siapkan import muat, pra_pohon

X, y, num = muat()
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                  random_state=7)
pra = pra_pohon(num).fit(Xl)
Zl, Zu = pra.transform(Xl), pra.transform(Xu)
kurva = {}
for nu in [1.0, 0.3, 0.05]:
    gb = GradientBoostingClassifier(n_estimators=600, learning_rate=nu,
                                    max_depth=3, subsample=0.8,
                                    random_state=0).fit(Zl, yl)
    ll = [log_loss(yu, p[:, 1]) for p in gb.staged_predict_proba(Zu)]
    kurva[nu] = ll
    b = int(np.argmin(ll))
    print(f"laju {nu:4.2f}: log loss uji minimum {ll[b]:.4f} pada "
          f"{b + 1:3d} pohon; setelah 600 pohon {ll[-1]:.4f}")
with open("gambar/bab12/laju.dat", "w") as f:
    f.write("m nu1 nu03 nu005\n")
    for m in list(range(1, 50)) + list(range(50, 601, 10)):
        f.write(f"{m} " + " ".join(f"{kurva[nu][m - 1]:.4f}"
                                   for nu in [1.0, 0.3, 0.05]) + "\n")
