import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = "kode/data/"
biji = pd.read_csv(DATA + "mutu_biji.csv")
rasa = ["aroma", "flavor", "aftertaste", "acidity", "body", "balance",
        "overall"]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
baris = []
for k in [1, 3, 5, 9, 15, 25, 35, 51, 75, 101, 151, 201, 301]:   #<1>
    r = {"k": k}
    for w in ["uniform", "distance"]:                            #<2>
        m = make_pipeline(StandardScaler(),
                          KNeighborsClassifier(k, weights=w))
        s = cross_validate(m, biji[rasa], biji["asal"], cv=cv,
                           return_train_score=True)
        r[w] = s["test_score"].mean()
        if w == "uniform":
            r["latih"] = s["train_score"].mean()
    baris.append(r)
tab = pd.DataFrame(baris)
print(tab.round(3).to_string(index=False))
tab.to_csv("gambar/bab09/pilih_k.dat", sep=" ", index=False)
