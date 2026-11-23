import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB

DATA = "kode/data/"
df = pd.read_csv(DATA + "pelanggan.csv").sample(400, random_state=0)  #<1>
print(pd.crosstab(df["tingkat"], df["churn"]))
X = df["tingkat"].map({"perunggu": 0, "perak": 1, "emas": 2}).to_frame()
for alfa in [1e-10, 1.0]:                                        #<2>
    nb = CategoricalNB(alpha=alfa).fit(X, df["churn"])
    p = np.exp(nb.feature_log_prob_[0][1])                       #<3>
    print(f"alfa={alfa:g}: P(tingkat|churn) perunggu {p[0]:.4f}, "
          f"perak {p[1]:.4f}, emas {p[2]:.4f}")
    x = pd.DataFrame({"tingkat": [2]})
    print(f"         P(churn | emas) = {nb.predict_proba(x)[0, 1]:.4f}")
