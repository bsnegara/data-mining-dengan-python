from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from siapkan import muat, pipa

X, y = muat()
print(f"porsi churn: {y.mean():.4f}")
Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3,           #<1>
                                  stratify=y, random_state=7)
model = {
    "tebak mayoritas": DummyClassifier(strategy="most_frequent"), #<2>
    "pohon penuh": DecisionTreeClassifier(random_state=0),
    "pohon kedalaman 5": DecisionTreeClassifier(max_depth=5,
                                                random_state=0)}
for nama, m in model.items():
    p = pipa(m, skala=False).fit(Xl, yl)
    print(f"{nama:18s} latih {p.score(Xl, yl):.3f}"              #<3>
          f"  uji {p.score(Xu, yu):.3f}")
