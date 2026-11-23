import numpy as np
import pandas as pd
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer,
                                   RobustScaler, StandardScaler)

DATA = "kode/data/"
df = pd.read_csv(DATA + "pelanggan_bersih.csv")
x = df[["nilai_90h"]] / 1000

hasil = pd.DataFrame({
    "asli": x["nilai_90h"],
    "minmax": MinMaxScaler().fit_transform(x).ravel(),           #<1>
    "z": StandardScaler().fit_transform(x).ravel(),              #<2>
    "robust": RobustScaler().fit_transform(x).ravel(),           #<3>
    "log": np.log1p(x["nilai_90h"]),                             #<4>
    "yeo": PowerTransformer().fit_transform(x).ravel(),          #<5>
})
print(hasil.describe().loc[["mean", "std", "min", "50%", "max"]]
      .round(2).to_string())
print(hasil.skew().round(2).to_string())
