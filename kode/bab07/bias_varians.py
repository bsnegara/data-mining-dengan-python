import pandas as pd
from sklearn.model_selection import StratifiedKFold, validation_curve
from sklearn.tree import DecisionTreeClassifier
from siapkan import muat, pipa

X, y = muat()
cv = StratifiedKFold(5, shuffle=True, random_state=0)
kedalaman = list(range(1, 16))
latih, valid = validation_curve(                                 #<1>
    pipa(DecisionTreeClassifier(random_state=0), skala=False), X, y,
    param_name="decisiontreeclassifier__max_depth",
    param_range=kedalaman, cv=cv, scoring="roc_auc")
tab = pd.DataFrame({"kedalaman": kedalaman,
                    "latih": latih.mean(1), "valid": valid.mean(1),
                    "sd": valid.std(1)})
print(tab.round(3).to_string(index=False))
terbaik = tab.loc[tab["valid"].idxmax()]                         #<2>
print(f"terbaik: kedalaman {int(terbaik['kedalaman'])}, "
      f"AUC valid {terbaik['valid']:.3f}")
tab.to_csv("gambar/bab07/kedalaman.dat", sep=" ", index=False)
