import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from siapkan import DATA, muat, pra_pohon

X, _, num = muat()
y = pd.read_csv(DATA + "pelanggan.csv")["belanja_bulan_depan"] / 1000
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, random_state=7)
pra = pra_pohon(num).fit(Xl)
Zl, Zu = pra.transform(Xl), pra.transform(Xu)

nu, M = 0.1, 300                                                 #<1>
Fl = np.full(len(yl), yl.mean())                                 #<2>
Fu = np.full(len(yu), yl.mean())
for m in range(1, M + 1):
    r = yl.to_numpy() - Fl                                       #<3>
    h = DecisionTreeRegressor(max_depth=3, random_state=0).fit(Zl, r)
    Fl += nu * h.predict(Zl)                                     #<4>
    Fu += nu * h.predict(Zu)
    if m in (1, 10, 50, 100, 300):
        print(f"M={m:3d}: MAE latih {mean_absolute_error(yl, Fl):6.1f}"
              f"  MAE uji {mean_absolute_error(yu, Fu):6.1f}")

sk = GradientBoostingRegressor(n_estimators=M, learning_rate=nu,
                               max_depth=3, random_state=0).fit(Zl, yl)
print(f"sklearn GradientBoostingRegressor: MAE uji "
      f"{mean_absolute_error(yu, sk.predict(Zu)):.1f}")          #<5>
