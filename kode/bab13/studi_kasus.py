import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from siapkan import DATA, KATEGORIK, NUMERIK

warnings.filterwarnings("ignore")
df = pd.read_csv(DATA + "pelanggan.csv")
X, y = df[NUMERIK + KATEGORIK], df["belanja_bulan_depan"] / 1000
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, random_state=7)
pra = lambda: ColumnTransformer(
    [("num", StandardScaler(), NUMERIK),
     ("kat", OneHotEncoder(handle_unknown="ignore"), KATEGORIK)])
model = {
    "linier": make_pipeline(pra(), LinearRegression()),
    "LightGBM": make_pipeline(pra(), lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=15,
        min_child_samples=40, verbose=-1, random_state=0)),
    "MLP (4,)": TransformedTargetRegressor(make_pipeline(
        pra(), MLPRegressor(hidden_layer_sizes=(4,), max_iter=500,
                            early_stopping=True, n_iter_no_change=20,
                            random_state=0)),
        transformer=StandardScaler())}
pred = {n: m.fit(Xl, yl).predict(Xu) for n, m in model.items()}
pred["PyTorch 2 kepala"] = np.load("kode/bab13/prediksi_torch.npy")  #<1>

tk = df.loc[Xu.index, "tingkat"]
print("model             MAE   RMSE  nol-salah  total(jt) galat-total")
for n, p in pred.items():
    p = np.clip(p, 0, None)                                      #<2>
    nol = ((p < 20) & (yu > 0)).mean()                           #<3>
    tot = p.sum() / 1000
    print(f"{n:16s} {mean_absolute_error(yu, p):5.1f} "
          f"{np.sqrt(mean_squared_error(yu, p)):6.1f} {nol:9.1%} "
          f"{tot:9.1f} {tot / (yu.sum() / 1000) - 1:+10.1%}")    #<4>
print(f"{'aktual':16s} {'':5s} {'':6s} {'':9s} {yu.sum() / 1000:9.1f}")

print("\ngalat per tingkat (MAE, ribu Rp):")                      #<5>
tab = pd.DataFrame({n: (np.abs(yu - np.clip(p, 0, None))
                        .groupby(tk).mean()) for n, p in pred.items()})
tab["n"] = tk.value_counts()
tab["rata2 aktual"] = yu.groupby(tk).mean()
print(tab.round(1).to_string())

gab = (pred["linier"] + pred["PyTorch 2 kepala"]) / 2            #<6>
print(f"\nrata-rata linier + PyTorch: MAE "
      f"{mean_absolute_error(yu, np.clip(gab, 0, None)):.1f}")
