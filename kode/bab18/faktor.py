import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from siapkan import muat

R, _, menu = muat()
X = np.log1p(R.to_numpy(float))                                  #<1>
for k in [1, 2, 3, 5, 10]:
    sv = TruncatedSVD(k, random_state=0).fit(X)
    print(f"k={k:2d}: varians terjelaskan {sv.explained_variance_ratio_.sum():.3f}")
sv = TruncatedSVD(2, random_state=0).fit(X)                      #<2>
Q = sv.components_.T * sv.singular_values_ / np.sqrt(len(X))
tab = pd.DataFrame(Q, index=menu["nama_menu"], columns=["f1", "f2"])
tab["kategori"] = menu["kategori"].to_numpy()
print(tab.sort_values("f2").round(3).to_string())
with open("gambar/bab18/faktor.dat", "w") as f:
    f.write("f1 f2 kat nama\n")
    kat = {"kopi susu": 0, "kopi hitam": 1, "non-kopi": 2, "makanan": 3}
    for n, r in tab.iterrows():
        f.write(f"{r.f1:.4f} {r.f2:.4f} {kat[r.kategori]} "
                f"{{{n}}}\n")
