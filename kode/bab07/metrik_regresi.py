import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from siapkan import DATA, muat, pipa

X, _ = muat()
y = pd.read_csv(DATA + "pelanggan.csv")["belanja_bulan_depan"] / 1000
print(f"target (ribu Rp): rata-rata {y.mean():.1f}, median "
      f"{y.median():.1f}, nol {(y == 0).mean():.1%}")
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, random_state=7)
model = {"rata-rata": DummyRegressor(),
         "linier": LinearRegression(),
         "hutan": RandomForestRegressor(300, min_samples_leaf=5,
                                        random_state=0, n_jobs=-1)}
print("model        MAE    RMSE    R2    MAPE*")
for nama, m in model.items():
    yp = pipa(m).fit(Xl, yl).predict(Xu)
    pos = yu > 0                                                 #<1>
    print(f"{nama:10s} {mean_absolute_error(yu, yp):6.1f} "
          f"{np.sqrt(mean_squared_error(yu, yp)):7.1f} "         #<2>
          f"{r2_score(yu, yp):5.3f} "
          f"{mean_absolute_percentage_error(yu[pos], yp[pos]):6.1%}")
