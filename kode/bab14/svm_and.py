import numpy as np
from sklearn.svm import SVC

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([-1, -1, -1, 1])                                    #<1>
svm = SVC(kernel="linear", C=1e6).fit(X, y)                      #<2>
w, b = svm.coef_[0], svm.intercept_[0]
print(f"w = {w.round(4)}, b = {b:.4f}")
print(f"lebar margin 2/||w|| = {2 / np.linalg.norm(w):.4f}")
print("vektor pendukung:", svm.support_vectors_.tolist())
print("alfa_i * y_i    :", svm.dual_coef_[0].round(4).tolist())  #<3>
print("y_i (w.x_i + b) :", (y * (X @ w + b)).round(4).tolist())  #<4>
