import time
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from siapkan import LAB, muat

df, _, _ = muat()
X, y = df[LAB + ["varietas", "proses"]], df["asal"]              #<1>
pra = ColumnTransformer([("num", StandardScaler(), LAB),
                         ("kat", OneHotEncoder(), ["varietas", "proses"])])
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for nama, m in {"satu-lawan-satu": SVC(C=10, gamma=0.05),        #<2>
                "satu-lawan-sisa": OneVsRestClassifier(
                    SVC(C=10, gamma=0.05))}.items():
    t0 = time.perf_counter()
    yp = cross_val_predict(make_pipeline(pra, m), X, y, cv=cv)
    print(f"{nama}: akurasi {(yp == y).mean():.3f}, "
          f"waktu {time.perf_counter() - t0:.1f} dtk")
kelas = sorted(y.unique())
print("matriks konfusi (satu-lawan-sisa), baris = asal sebenarnya")
print("        " + " ".join(f"{k[:7]:>7s}" for k in kelas))
for k, r in zip(kelas, confusion_matrix(y, yp, labels=kelas)):
    print(f"{k[:7]:7s} " + " ".join(f"{v:7d}" for v in r))
