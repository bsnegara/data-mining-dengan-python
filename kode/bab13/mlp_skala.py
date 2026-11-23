import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sys
sys.path.insert(0, "kode/bab12")
from siapkan import DATA, KATEGORIK, NUMERIK

df = pd.read_csv(DATA + "pelanggan.csv")
X, y = df[NUMERIK + KATEGORIK], df["belanja_bulan_depan"] / 1000
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, random_state=7)


def pipa(skala, lapisan, alpha=1e-4, solver="adam", skala_y=True):
    num = StandardScaler() if skala else "passthrough"
    m = make_pipeline(ColumnTransformer(
        [("num", num, NUMERIK),
         ("kat", OneHotEncoder(handle_unknown="ignore"), KATEGORIK)]),
        MLPRegressor(hidden_layer_sizes=lapisan, alpha=alpha, solver=solver,
                     learning_rate_init=1e-3,
                     max_iter=500, early_stopping=True,          #<1>
                     n_iter_no_change=20, random_state=0))
    if not skala_y:
        return m
    return TransformedTargetRegressor(m, transformer=StandardScaler())


for solver, skala, sy in [("sgd", False, False), ("sgd", True, False),
                          ("sgd", True, True), ("adam", True, True)]:  #<2>
    m = pipa(skala, (32,), solver=solver, skala_y=sy).fit(Xl, yl)
    print(f"{solver:4s} skala X={skala!s:5s} skala y={sy!s:5s}: MAE uji "
          f"{mean_absolute_error(yu, m.predict(Xu)):7.1f}")
print()
for lap in [(4,), (32,), (128,), (64, 32), (128, 64, 32)]:       #<3>
    m = pipa(True, lap).fit(Xl, yl)
    mlp = m.regressor_[-1]
    n = sum(w.size for w in mlp.coefs_) + sum(
        b.size for b in mlp.intercepts_)
    print(f"lapisan {str(lap):13s} parameter {n:6d}  MAE uji "
          f"{mean_absolute_error(yu, m.predict(Xu)):5.1f}")

m = pipa(True, (64, 32)).fit(Xl, yl)
with open("gambar/bab13/kurva_latih.dat", "w") as f:             #<4>
    f.write("epoch loss r2val\n")
    mlp = m.regressor_[-1]
    for i, (l, v) in enumerate(zip(mlp.loss_curve_,
                                   mlp.validation_scores_), 1):
        f.write(f"{i} {l:.4f} {v:.4f}\n")
