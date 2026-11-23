import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = "kode/data/"
biji = pd.read_csv(DATA + "mutu_biji.csv")
fisik = ["kadar_air", "cacat_primer", "cacat_sekunder", "ukuran_screen",
         "densitas_gl", "ketinggian_mdpl", "lama_simpan_hari"]
X, y = biji[fisik], biji["skor_total"]                           #<1>
cv = KFold(5, shuffle=True, random_state=0)
model = {"rata-rata": DummyRegressor(),
         "linier": make_pipeline(StandardScaler(), LinearRegression())}
for k in [1, 5, 10, 20, 40]:
    model[f"kNN k={k}"] = make_pipeline(StandardScaler(),
                                        KNeighborsRegressor(k))  #<2>
model["kNN k=20 jarak"] = make_pipeline(
    StandardScaler(), KNeighborsRegressor(20, weights="distance"))
for nama, m in model.items():
    mae = -cross_val_score(m, X, y, cv=cv,
                           scoring="neg_mean_absolute_error").mean()
    print(f"{nama:15s} MAE {mae:.2f} poin")
