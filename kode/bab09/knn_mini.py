import numpy as np
import pandas as pd

mini = pd.DataFrame({
    "acidity": [7.75, 7.50, 7.75, 6.75, 6.50, 7.00, 7.25, 6.75],
    "body":    [7.00, 7.25, 7.50, 7.75, 7.50, 7.75, 7.00, 7.00],
    "asal":    ["Kerinci", "Kerinci", "Toraja", "Gayo", "Gayo", "Gayo",
                "Kerinci", "Flores"]})
baru = np.array([7.25, 7.40])                                    #<1>
selisih = mini[["acidity", "body"]].to_numpy() - baru
mini["jarak"] = np.sqrt((selisih ** 2).sum(axis=1)).round(3)     #<2>
mini = mini.sort_values("jarak")
print(mini.to_string())
for k in [1, 3, 5]:                                              #<3>
    suara = mini.head(k)["asal"].value_counts()
    print(f"k={k}:", ", ".join(f"{a} {n}" for a, n in suara.items()))
