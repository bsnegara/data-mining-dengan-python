import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

DATA = "kode/data/"
biji = pd.read_csv(DATA + "mutu_biji.csv")
biji["full_washed"] = (biji["proses"] == "full washed").astype(int)
fitur = ["ketinggian_mdpl", "cacat_primer", "cacat_sekunder",
         "kadar_air", "jamur", "full_washed"]
Xl, Xu, yl, yu = train_test_split(biji[fitur], biji["skor_total"],
                                  test_size=.3, random_state=0)
print("MAE tebak rata-rata:",
      round(mean_absolute_error(yu, [yl.mean()] * len(yu)), 2))   #<1>
for d in [1, 2, 3, 5, 8]:
    m = DecisionTreeRegressor(max_depth=d, min_samples_leaf=20,  #<2>
                              random_state=0).fit(Xl, yl)
    print(f"kedalaman {d}: MAE uji = "
          f"{mean_absolute_error(yu, m.predict(Xu)):.2f}")
m = DecisionTreeRegressor(max_depth=2, min_samples_leaf=20).fit(Xl, yl)
print(export_text(m, feature_names=fitur, decimals=1))           #<3>
